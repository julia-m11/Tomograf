import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid
import datetime
import numpy as np
from skimage.util import img_as_ubyte
from skimage.exposure import rescale_intensity

def save_as_dicom(file_name, img, patient_data):
    img_rescaled = rescale_intensity(img, in_range='image', out_range=(0, 1))
    img_converted = img_as_ubyte(img_rescaled)
    pixel_bytes = img_converted.tobytes()
    if len(pixel_bytes) % 2 != 0:
        pixel_bytes += b'\x00'

    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.ImplementationClassUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian 

    ds = FileDataset(file_name, {}, file_meta=file_meta, preamble=b"\0" * 128)

    ds.PatientName = patient_data.get("PatientName", "Unknown^Patient")
    ds.PatientID = patient_data.get("PatientID", "000000")
    ds.ImageComments = patient_data.get("ImageComments", "000000")
    ds.ContentDate = datetime.datetime.now().strftime('%Y%m%d')
    # ds.ContentTime = datetime.datetime.now().strftime('%H%M%S.%f')
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = SecondaryCaptureImageStorage
    ds.Modality = "CT"

    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows, ds.Columns = img_converted.shape
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0 
    
    ds.ImagePositionPatient = [0, 0, 0]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.WindowCenter = "127"
    ds.WindowWidth = "255"
    ds.RescaleIntercept = "0"
    ds.RescaleSlope = "1"

    ds.PixelData = pixel_bytes

    pydicom.dataset.validate_file_meta(ds.file_meta, enforce_standard=True)
    ds.save_as(file_name, write_like_original=False)
    print(f"DICOM saved: {file_name}")