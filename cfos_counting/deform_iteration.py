from deform import *
import os
import shutil
from tifffile import imsave, imread
import SimpleITK as sitk
import multiprocessing

input_dir = "F:/chaoyu/test/thy1"
output_dir = "F:/chaoyu/test/average_thy1_brain"

SUFFIX = "result.tif"
PARAMETERS_1 = [join(os.path.dirname(__file__), "parameters_Rigid.txt"),
                join(os.path.dirname(__file__), "parameters_Affine.txt"),
                join(os.path.dirname(__file__), "parameters_BSpline_3d_alt.txt")]
PARAMETERS_2 = [join(os.path.dirname(__file__), "parameters_BSpline_3d.txt")]
params1 = sitk.VectorOfParameterMap()
for f in PARAMETERS_1:
    params1.append(sitk.ReadParameterFile(f))
params2 = sitk.VectorOfParameterMap()
for f in PARAMETERS_2:
    params2.append(sitk.ReadParameterFile(f))
FIRST_TEMPLETE = "F:/chaoyu/test/thy1/template.tif"
ITERATIONS = 4
SHAPE = (456, 320, 528)


def align_brain(args):
    f = args[0]
    iteration_dir = args[1]
    templete = args[2]
    if os.path.exists(join(iteration_dir, f + ".mha")):
        return
    templete = sitk.GetImageFromArray(templete)
    elastix_ = sitk.ElastixImageFilter()
    elastix_.SetParameterMap(params1)
    elastix_.SetFixedImage(templete)
    elastix_.SetMovingImage(sitk.ReadImage(join(input_dir, f, SUFFIX)))
    elastix_.Execute()
    sitk.WriteImage(elastix_.GetResultImage(), join(iteration_dir, f + ".mha"))


def editMHDFile(srcfile, dstfile, new_path):
    src = open(srcfile, "r")
    dst = open(dstfile, "w")
    while 1:
        line = src.readline()
        if len(line) == 0:
            break
        l = line.split(' ')
        if l[0] == "ElementDataFile":
            line = "ElementDataFile = " + new_path + ".raw\n"
        dst.write(line)
    src.close()
    dst.close()

if __name__ == "__main__":
    brains = [f
              for f in os.listdir(input_dir)
              if os.path.exists(join(input_dir, f, SUFFIX))]
    first_template = sitk.ReadImage(FIRST_TEMPLETE)
    template = first_template

    elastix_ = sitk.ElastixImageFilter()
    pool = multiprocessing.Pool()
    for i in range(ITERATIONS):
        iteration_dir = join(output_dir, "iteration" + str(i))
        if not os.path.exists(iteration_dir):
            os.mkdir(iteration_dir)
        elastix_.SetOutputDirectory(iteration_dir)

        templete_ = sitk.GetArrayFromImage(template)
        param_brains = [[f, iteration_dir, templete_] for f in brains]
        pool.map(align_brain, param_brains, 1)

        if not os.path.exists(join(iteration_dir, "average.tif")):
            brain_images = []
            weights = []
            for f in brains:
                brain_image = sitk.ReadImage(join(iteration_dir, f + ".mha"))
                weight = sitk.BinaryThreshold(brain_image, 2, 255)
                brain_image = sitk.GetArrayFromImage(brain_image)
                brain_images.append(brain_image)
                weight = sitk.GetArrayFromImage(weight)
                weights.append(weight)
            average = np.sum(brain_images, axis=0)
            weights = np.sum(weights, axis=0) + 1
            average = average / weights
            average = np.float32(average)
            imsave(join(iteration_dir, "average.tif"), average)
            template = sitk.GetImageFromArray(average)
        else:
            template = imread(join(iteration_dir, "average.tif"))
            template = np.float32(template)
            template = sitk.GetImageFromArray(template)
'''
        if not os.path.exists(join(iteration_dir, "templete.mha")):
            elastix_.SetParameterMap(params2)
            elastix_.SetFixedImage(first_template)
            elastix_.SetMovingImage(template)
            elastix_.Execute()
            template = elastix_.GetResultImage()
            sitk.WriteImage(template, join(iteration_dir, "templete.mha"))
'''
