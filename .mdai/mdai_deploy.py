import os
import dicom2nifti
import pydicom
from io import BytesIO

from z_unet import unet

import nibabel
import numpy as np
import tensorflow as tf
from skimage.measure import find_contours
from dicom2nifti.image_volume import ImageVolume, SliceType


class MDAIModel:
    def __init__(self):
        modelpath = os.path.join(os.path.dirname(__file__), "../weights.hdf5")
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

    def load(self, nifti_image):
        return ImageVolume(nifti_image)

    def reorient_image(self, input_image):
        """
        Change the orientation of the Image data in order to be in LAS space
        x will represent the coronal plane, y the sagittal and z the axial plane.
        x increases from Right (R) to Left (L), y from Posterior (P) to Anterior (A) and z from Inferior (I) to Superior (S)

        :returns: The output image in nibabel form
        :param output_image: filepath to the nibabel image
        :param input_image: filepath to the nibabel image
        """
        # Use the imageVolume module to find which coordinate corresponds to each plane
        # and get the image data in RAS orientation
        # print 'Reading nifti'
        image = self.load(input_image)

        # 4d have a different conversion to 3d
        # print 'Reorganizing data'
        if image.nifti_data.squeeze().ndim == 4:
            new_image = self._reorient_4d(image)
        elif image.nifti_data.squeeze().ndim == 3:
            new_image = self._reorient_3d(image)
        else:
            raise Exception("Only 3d and 4d images are supported")

        # print 'Recreating affine'
        affine = image.nifti.affine
        # Based on VolumeImage.py where slice orientation 1 represents the axial plane
        # Flipping on the data may be needed based on x_inverted, y_inverted, ZInverted

        # Create new affine header by changing the order of the columns of the input image header
        # the last column with the origin depends on the origin of the original image, the size and the direction of x,y,z

        new_affine = np.eye(4)
        new_affine[:, 0] = affine[:, image.sagittal_orientation.normal_component]
        new_affine[:, 1] = affine[:, image.coronal_orientation.normal_component]
        new_affine[:, 2] = affine[:, image.axial_orientation.normal_component]
        point = [0, 0, 0, 1]

        # If the orientation of coordinates is inverted, then the origin of the "new" image
        # would correspond to the last voxel of the original image
        # First we need to find which point is the origin point in image coordinates
        # and then transform it in world coordinates
        if not image.axial_orientation.x_inverted:
            new_affine[:, 0] = -new_affine[:, 0]
            point[image.sagittal_orientation.normal_component] = (
                image.dimensions[image.sagittal_orientation.normal_component] - 1
            )
            # new_affine[0, 3] = - new_affine[0, 3]
        if image.axial_orientation.y_inverted:
            new_affine[:, 1] = -new_affine[:, 1]
            point[image.coronal_orientation.normal_component] = (
                image.dimensions[image.coronal_orientation.normal_component] - 1
            )
            # new_affine[1, 3] = - new_affine[1, 3]
        if image.coronal_orientation.y_inverted:
            new_affine[:, 2] = -new_affine[:, 2]
            point[image.axial_orientation.normal_component] = (
                image.dimensions[image.axial_orientation.normal_component] - 1
            )
            # new_affine[2, 3] = - new_affine[2, 3]

        new_affine[:, 3] = np.dot(affine, point)

        # DONE: Needs to update new_affine, so that there is no translation difference between the original
        # and created image (now there is 1-2 voxels translation)
        new_nifti_image = nibabel.Nifti1Image(new_image, new_affine)
        return new_nifti_image

    def _reorient_4d(self, image):
        """
        Reorganize the data for a 4d nifti
        """
        # print 'converting 4d image'
        # Create empty array where x,y,z correspond to LR (sagittal), PA (coronal), IS (axial) directions and the size
        # of the array in each direction is the same with the corresponding direction of the input image.
        new_image = np.zeros(
            [
                image.dimensions[image.sagittal_orientation.normal_component],
                image.dimensions[image.coronal_orientation.normal_component],
                image.dimensions[image.axial_orientation.normal_component],
                image.dimensions[3],
            ],
            dtype=image.nifti_data.dtype,
        )

        # loop over all timepoints
        for timepoint in range(0, image.dimensions[3]):
            # Fill the new image with the values of the input image but with mathicng the orientation with x,y,z
            if image.coronal_orientation.y_inverted:
                for i in range(new_image.shape[2]):
                    new_image[:, :, i, timepoint] = np.fliplr(
                        np.squeeze(
                            image.get_slice(
                                SliceType.AXIAL, new_image.shape[2] - 1 - i, timepoint
                            ).original_data
                        )
                    )
            else:
                for i in range(new_image.shape[2]):
                    new_image[:, :, i, timepoint] = np.fliplr(
                        np.squeeze(
                            image.get_slice(SliceType.AXIAL, i, timepoint).original_data
                        )
                    )

        return new_image

    def _reorient_3d(self, image):
        """
        Reorganize the data for a 3d nifti
        """
        # Create empty array where x,y,z correspond to LR (sagittal), PA (coronal), IS (axial) directions and the size
        # of the array in each direction is the same with the corresponding direction of the input image.
        new_image = np.zeros(
            [
                image.dimensions[image.sagittal_orientation.normal_component],
                image.dimensions[image.coronal_orientation.normal_component],
                image.dimensions[image.axial_orientation.normal_component],
            ],
            dtype=image.nifti_data.dtype,
        )

        # Fill the new image with the values of the input image but with matching the orientation with x,y,z
        if image.coronal_orientation.y_inverted:
            for i in range(new_image.shape[2]):
                new_image[:, :, i] = np.fliplr(
                    np.squeeze(
                        image.get_slice(
                            SliceType.AXIAL, new_image.shape[2] - 1 - i
                        ).original_data
                    )
                )
        else:
            for i in range(new_image.shape[2]):
                new_image[:, :, i] = np.fliplr(
                    np.squeeze(image.get_slice(SliceType.AXIAL, i).original_data)
                )

        return new_image

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
            dicom_files, output_file=None, reorient_nifti=False
        )

        print("Converted to NIFTI", flush=True)

        test_images = self.reorient_image(nifti_file["NII"])
        test_images = test_images.get_fdata()
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
                        "class_index": label,
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
        return outputs

