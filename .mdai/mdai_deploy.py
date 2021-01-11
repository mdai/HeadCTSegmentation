import os
import dicom2nifti
import pydicom
from io import BytesIO

from z_unet_new import unet

import tempfile
import shutil
import nibabel
import numpy as np
import tensorflow as tf
from skimage.measure import find_contours
from dicom2nifti.image_volume import ImageVolume, SliceType


class MDAIModel:
    def __init__(self):
        modelpath = os.path.join(os.path.dirname(__file__), "../weights.hdf5")
        self.tempdir = tempfile.mkdtemp()
        self.nb_classes = 17

        self.model = unet(self.nb_classes)
        self.model.load_weights(modelpath)

    def sort_dicoms(self, dicoms):
        """
        Sort the dicoms based om the image possition patient

        :param dicoms: list of dicoms
        """
        # find most significant axis to use during sorting
        # the original way of sorting (first x than y than z) does not work in certain border situations
        # where for exampe the X will only slightly change causing the values to remain equal on multiple slices
        # messing up the sorting completely)
        dicom_input_sorted_x = sorted(dicoms, key=lambda x: (x.ImagePositionPatient[0]))
        dicom_input_sorted_y = sorted(dicoms, key=lambda x: (x.ImagePositionPatient[1]))
        dicom_input_sorted_z = sorted(dicoms, key=lambda x: (x.ImagePositionPatient[2]))
        diff_x = abs(
            dicom_input_sorted_x[-1].ImagePositionPatient[0]
            - dicom_input_sorted_x[0].ImagePositionPatient[0]
        )
        diff_y = abs(
            dicom_input_sorted_y[-1].ImagePositionPatient[1]
            - dicom_input_sorted_y[0].ImagePositionPatient[1]
        )
        diff_z = abs(
            dicom_input_sorted_z[-1].ImagePositionPatient[2]
            - dicom_input_sorted_z[0].ImagePositionPatient[2]
        )
        if diff_x >= diff_y and diff_x >= diff_z:
            return dicom_input_sorted_x
        if diff_y >= diff_x and diff_y >= diff_z:
            return dicom_input_sorted_y
        if diff_z >= diff_x and diff_z >= diff_y:
            return dicom_input_sorted_z

    def predict(self, data):
        input_files = data["files"]
        input_annotations = data["annotations"]
        input_args = data["args"]

        outputs = []

        dicom_files = []
        for file in input_files:
            if file["content_type"] != "application/dicom":
                continue

            dicom_files.append(pydicom.dcmread(BytesIO(file["content"])))

        dicom_files = self.sort_dicoms(dicom_files)

        nifti_file = dicom2nifti.convert_dicom.dicom_array_to_nifti(
            dicom_files,
            output_file=os.path.join(self.tempdir, "input.nii.gz"),
            reorient_nifti=True,
        )
        print("Converted to NIFTI", flush=True)

        test_images = nibabel.load(
            os.path.join(self.tempdir, "input.nii.gz")
        ).get_fdata()
        test_images = np.moveaxis(test_images, -1, 0)
        test_images = np.expand_dims(test_images, -1).astype(np.float32)
        num_images = test_images.shape[0]
        print("Running model...", flush=True)
        predlabel = self.model.predict(test_images, batch_size=1, verbose=1)
        print("Model run completed", flush=True)
        predlabel = predlabel.reshape((num_images, 512, 512, self.nb_classes))
        predlabel = np.argmax(predlabel, axis=3)
        predlabel = np.moveaxis(predlabel, 0, -1).astype("uint16")
        # predlabel = np.flip(predlabel, 0)
        # predlabel = np.flip(predlabel, 1)
        # predlabel = np.flip(predlabel, 2)

        for i in range(num_images):
            vals = set(np.unique(predlabel[:, :, i]))
            masks = [
                (np.uint8(np.rot90(predlabel[:, :, i]) == t), t)
                for t in range(1, self.nb_classes)
                if t in vals
            ]

            if masks:
                contours = [
                    (submask, m[1]) for m in masks for submask in find_contours(m[0], 0)
                ]
                # contours = [(find_contours(m[0], 0)[0], m[1]) for m in masks]
                preds = []
                for contour, label in contours:
                    data = {"vertices": [[(v[1]), (v[0])] for v in contour.tolist()]}
                    output = {
                        "type": "ANNOTATION",
                        "study_uid": str(dicom_files[i].StudyInstanceUID),
                        "series_uid": str(dicom_files[i].SeriesInstanceUID),
                        "instance_uid": str(dicom_files[i].SOPInstanceUID),
                        "class_index": int(label),
                        "data": data,
                    }
                    preds.append(output)
            else:
                preds = [
                    {
                        "type": "NONE",
                        "study_uid": str(dicom_files[i].StudyInstanceUID),
                        "series_uid": str(dicom_files[i].SeriesInstanceUID),
                        "instance_uid": str(dicom_files[i].SOPInstanceUID),
                    }
                ]
            outputs += preds
        shutil.rmtree(self.tempdir)
        return outputs

