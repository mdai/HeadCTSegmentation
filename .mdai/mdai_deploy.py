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

import cv2
from dicom2nifti.image_volume import ImageVolume, SliceType
import dicom2nifti.settings as settings

settings.disable_validate_orthogonal()
settings.disable_validate_slice_increment()


class MDAIModel:
    def __init__(self):
        modelpath = os.path.join(os.path.dirname(__file__), "../weights.hdf5")
        self.tempdir = tempfile.mkdtemp()
        self.nb_classes = 17

        self.model = unet(self.nb_classes)
        self.model.load_weights(modelpath)

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

        dicom_files = dicom2nifti.common.sort_dicoms(dicom_files)

        nifti_file = dicom2nifti.convert_dicom.dicom_array_to_nifti(
            dicom_files,
            output_file=os.path.join(
                self.tempdir, dicom_files[0].SeriesInstanceUID + ".nii.gz"
            ),
            reorient_nifti=True,
        )
        print("Converted to NIFTI", flush=True)

        test_images = nibabel.load(
            os.path.join(self.tempdir, dicom_files[0].SeriesInstanceUID + ".nii.gz")
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

        for i in range(num_images):
            vals = set(np.unique(predlabel[:, :, i]))
            masks = [
                (np.uint8(np.rot90(predlabel[:, :, i]) == t), t)
                for t in range(1, self.nb_classes)
                if t in vals
            ]

            if masks:
                preds = []
                for submask, label in masks:
                    output = {
                        "type": "ANNOTATION",
                        "study_uid": str(dicom_files[i].StudyInstanceUID),
                        "series_uid": str(dicom_files[i].SeriesInstanceUID),
                        "instance_uid": str(dicom_files[i].SOPInstanceUID),
                        "class_index": int(label),
                        "data": {"mask": submask.tolist()},
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

        os.remove(
            os.path.join(self.tempdir, dicom_files[0].SeriesInstanceUID + ".nii.gz")
        )
        return outputs

