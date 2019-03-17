import os
import torch, cv2
import SimpleITK as sitk
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

k_surface_band = cv2.getGaussianKernel(3, -1)
k_surface_band = np.matmul(k_surface_band, np.transpose(k_surface_band))
k_surface_band[1, 1] = k_surface_band[1, 1] - np.sum(k_surface_band)
k_surface_band = np.array([[k_surface_band]])
k_surface_band = torch.FloatTensor(k_surface_band)
k_surface_band = Variable(k_surface_band)
k_surface_band_c = k_surface_band.cuda()

k_grad = np.float32([[[0.5, 1, 2, 4, 8, 0, -8, -4, -2, -1, 0.5]]])
k_grad = np.transpose(k_grad, [2, 1, 0])
k_grad = sitk.GetImageFromArray(k_grad)

k_grad2 = np.float32([[[1, 0, -1]]])
k_grad2 = np.transpose(k_grad2, [2, 1, 0])
k_grad2 = sitk.GetImageFromArray(k_grad2)

def fill_blank(img):
    state = sitk.Image([img.GetSize()[0], img.GetSize()[1]], sitk.sitkUInt16) * 0
    out_ = []
    for i in range(img.GetSize()[2]):
        sl = sitk.Extract(img, [img.GetSize()[0], img.GetSize()[1], 0], [0, 0, i])
        sl.SetOrigin([0., 0.])
        sl.SetSpacing([1, 1])
        zero_area = sitk.LessEqual(sl, 1)
        zero_area = sitk.Cast(zero_area, sitk.sitkUInt16)
        state = state * zero_area + sl * sitk.Not(zero_area)
        out_.append(state)
    out = []
    state = sitk.Image([img.GetSize()[0], img.GetSize()[1]], sitk.sitkUInt16) * 0
    for i in range(img.GetSize()[2] - 1, -1, -1):
        sl = out_[i]
        sl.SetOrigin([0., 0.])
        sl.SetSpacing([1, 1])
        zero_area = sitk.LessEqual(sl, 1)
        zero_area = sitk.Cast(zero_area, sitk.sitkUInt16)
        state = state * zero_area + sl * sitk.Not(zero_area)
        out.append(state)
    out.reverse()
    return sitk.JoinSeries(out)


def calc_surface_height_map(img: sitk.Image):
    img.SetSpacing([1, 1, 1])
    img.SetOrigin([0, 0, 0])
    src_img = sitk.GetArrayFromImage(img)
    src_img = torch.FloatTensor(np.float32(src_img))
    size_ = img.GetSize()
    tf = sitk.AffineTransform(3)
    tf.Scale([2, 2, 1])
    proc_size = [int(size_[0] / 2), int(size_[1] / 2), size_[2]]
    img = sitk.Resample(img, proc_size, tf)
    gpu_load_grad = False
    if proc_size[0] * proc_size[1] * proc_size[2] < 900000000:
        gpu_load_grad = True
    img = fill_blank(img)
    img = sitk.Cast(img, sitk.sitkFloat32)
    img = (sitk.Log(img) - 4.6) * 39.4
    img = sitk.Clamp(img, sitk.sitkFloat32, 0, 255)
    def get_edge_grad(img_: sitk.Image, ul):
        grad_m = sitk.Convolution(img_, k_grad)
        if ul == 1:
            grad_m = sitk.Clamp(grad_m, sitk.sitkFloat32, 0, 65535)
        else:
            grad_m = sitk.Clamp(grad_m, sitk.sitkFloat32, -65535, 0)
        grad_m = ul * sitk.Convolution(grad_m, k_grad2)
        grad_m = sitk.GetArrayFromImage(grad_m)
        grad_m = torch.FloatTensor(grad_m)
        if gpu_load_grad:
            grad_m = grad_m.cuda()
        return grad_m
    u_grad_m = get_edge_grad(img, 1)
    l_grad_m = get_edge_grad(img, -1)
    img = torch.Tensor(sitk.GetArrayFromImage(img)).byte()
    #if gpu_load_grad:
    img = img.cuda()
    u = (torch.rand(1, 1, u_grad_m.shape[1], u_grad_m.shape[2]) * (u_grad_m.shape[0]) / 2) + l_grad_m.shape[0] / 2
    l = (l_grad_m.shape[0] / 1 + torch.rand(1, 1, l_grad_m.shape[1], l_grad_m.shape[2]) * (l_grad_m.shape[0]) / 2)

    lr = 0.002
    momentum = 0.9
    lr_decay = 0.0002
    u_grad = torch.zeros(1, 1, u_grad_m.shape[1], u_grad_m.shape[2])
    l_grad = torch.zeros(1, 1, l_grad_m.shape[1], l_grad_m.shape[2])
    k = k_surface_band
    #if gpu_load_grad:
    u = u.cuda()
    l = l.cuda()
    u_grad = u_grad.cuda()
    l_grad = l_grad.cuda()
    k = k_surface_band_c
    gu, gl = None, None
    for i in range(8000):
        def calc_grad(s, grad, grad_m, ul):
            g = torch.clamp(s[0], 0, grad_m.shape[0] - 1).long()
            u_plane_bend = F.pad(s, (1, 1, 1, 1), mode='reflect')
            u_plane_bend = F.conv2d(u_plane_bend, k, padding=0)[0].data
            if gpu_load_grad:
                u_edge = torch.gather(grad_m, 0, g)
            else:
                u_edge = torch.gather(grad_m, 0, g.cpu()).cuda()
            grad = u_edge + \
                   3000 * u_plane_bend + \
                   0.05 * ul * torch.clamp(l - u - 75, -1000, 1) + \
                   momentum * grad
            return grad, g
        u_grad, gu = calc_grad(u, u_grad, u_grad_m, 1)
        l_grad, gl = calc_grad(l, l_grad, l_grad_m, -1)
        u += lr * u_grad
        l += lr * l_grad
        u = torch.clamp(u, 0, u_grad_m.shape[0] - 1)
        l = torch.clamp(l, 0, u_grad_m.shape[0] - 1)
        lr *= (1 - lr_decay)
        #v = torch.gather(img, 0, gu)[0].cpu().numpy()
        #print(torch.mean(u))
        #if i % 10 == 0:
        #    cv2.imshow('s', np.uint8(v * 5))
        #    cv2.waitKey(1)
        print(i)

    umap = np.float32((u + 1).cpu().numpy()[0][0])
    lmap = np.float32((l - 1).cpu().numpy()[0][0])
    umap = cv2.resize(umap, (size_[0], size_[1]))
    lmap = cv2.resize(lmap, (size_[0], size_[1]))
    u_surface = torch.gather(src_img, 0, torch.Tensor(np.array([umap])).long())[0].cpu().numpy()
    l_surface = torch.gather(src_img, 0, torch.Tensor(np.array([lmap])).long())[0].cpu().numpy()
    u_surface = np.clip((np.log(u_surface) - 4.6) * 39.4, 0, 255)
    l_surface = np.clip((np.log(l_surface) - 4.6) * 39.4, 0, 255)

    umap = sitk.GetImageFromArray(umap)
    lmap = sitk.GetImageFromArray(lmap)
    u_surface = sitk.GetImageFromArray(np.uint8(u_surface))
    l_surface = sitk.GetImageFromArray(np.uint8(l_surface))

    return umap, lmap, u_surface, l_surface


def get_image(src, umap, lmap, thickness, scale=6.326):
    zeros = sitk.Image(umap.GetSize(), sitk.sitkFloat64) * 0
    lmap -= thickness
    u_transform = sitk.Compose(zeros, zeros,
                               sitk.Cast(umap, sitk.sitkFloat64))
    l_transform = sitk.Compose(zeros, zeros,
                               sitk.Cast(lmap, sitk.sitkFloat64))

    size = np.array(src.GetSize())
    size = np.int32(size / scale).tolist()
    size[2] = thickness
    print(size)
    spacing = [scale, scale, 1]
    df = sitk.JoinSeries([u_transform, l_transform])
    tf = sitk.ScaleTransform(3, [1., 1., 1 / thickness])
    df = sitk.Resample(df, [df.GetSize()[0], df.GetSize()[1], thickness], tf, sitk.sitkLinear)
    tf = sitk.DisplacementFieldTransform(sitk.Image(df))
    #sitk.WriteImage(sitk.VectorIndexSelectionCast(df, 2), dst + 's.mhd')
    out = sitk.Resample(src, size, tf, sitk.sitkLinear, [0., 0., 0.], spacing)
    return out


def load_image(srcdir):
    imgfiles = [f for f in os.listdir(os.path.join(srcdir, '8')) if f.split('.')[-1] == 'tif']
    imgfiles.sort(key=lambda x: int(x.split('.')[0]))
    imgfiles = [os.path.join(srcdir, '8', f) for f in imgfiles]
    img = []
    im = cv2.imread(imgfiles[0], -1)
    shape = im.shape
    for f in imgfiles:
        im = cv2.imread(f, -1)
        if shape[0] != im.shape[0] or shape[1] != im.shape[1]:
            break
        img.append(im)
    img = np.array(img)
    return img


def brightness_correct_h(img, k, b, co):
    res = []
    for i in range(img.shape[0]):
        a = co[(i - b) % k]
        res.append(cv2.subtract(img[i], 100) / a + 100)
    res = np.array(res)
    return res


def brightness_correct_v(img, co):
    res = []
    img = sitk.Cast(img, sitk.sitkFloat32)
    img = sitk.GetArrayFromImage(img)
    for i in range(img.shape[0]):
        a = co[i]
        res.append(np.clip(cv2.subtract(img[i], 100) / a + 100, 0, 65535))
    res = np.uint16(res)
    res = sitk.GetImageFromArray(res)
    return res


h_correction = cv2.imread('h_correction.tif', -1)
h_correction = h_correction[:,0]
v_correction = cv2.imread('v_correction.tif', -1)
v_correction = v_correction[:,0]


def flatten_fix(srcdir):
    img = load_image(srcdir)
    img = brightness_correct_h(img, 227, 16, h_correction)
    img = np.transpose(img, [1, 0, 2])
    img = sitk.GetImageFromArray(img)
    img = sitk.Cast(img, sitk.sitkUInt16)
    umap, lmap, u_surface, l_surface = calc_surface_height_map(img)
    umap += 1
    lmap += 1
    #a = sitk.GetArrayFromImage(umap)
    #cv2.imwrite(os.path.join(srcdir, 'a.tif'), np.uint8(a))
    #u = sitk.GetArrayFromImage(lmap)
    #cv2.imwrite(os.path.join(srcdir, 'u.tif'), np.uisnt8(u))
    sitk.WriteImage(sitk.Cast(umap, sitk.sitkUInt8), os.path.join(srcdir, 'a.tif'))
    sitk.WriteImage(sitk.Cast(lmap, sitk.sitkUInt8), os.path.join(srcdir, 'u.tif'))
    img = get_image(img, umap, lmap, 13)
    img = brightness_correct_v(img, v_correction)
    wr = sitk.ImageSeriesWriter()
    wr.SetFileNames([os.path.join(srcdir, '25/{0}.tif'.format(i)) for i in range(13)])
    wr.Execute(img)


def select_slice(srcdir):
    img = load_image(srcdir)
    img = np.transpose(img, [1, 0, 2])
    img = np.uint16(img)
    umap = sitk.ReadImage(os.path.join(srcdir, 'a.tif'))
    lmap = sitk.ReadImage(os.path.join(srcdir, 'u.tif'))
    umap = sitk.Compose(sitk.Image(umap.GetSize(), sitk.sitkFloat64),
                          sitk.Image(umap.GetSize(), sitk.sitkFloat64),
                          sitk.Cast(umap, sitk.sitkFloat64))
    lmap = sitk.Compose(sitk.Image(lmap.GetSize(), sitk.sitkFloat64),
                          sitk.Image(lmap.GetSize(), sitk.sitkFloat64),
                          sitk.Cast(lmap, sitk.sitkFloat64))
    img = get_image(img, umap, lmap, 75, 1)
    idx = np.random.randint(0, 12)
    sl = img[:,:,int(idx * 5.36 + 2.68)]
    return idx, sl


if __name__ == '__main__':
    src = 'D:/Hao/reconstructed/cfos_counting/cfos_1736/cfos_1736/x'
    #dst = 'D:/chaoyu/cfos/slice/cfos-FS1-407_488'
    dir_list = [f for f in os.listdir(src) if os.path.isdir(os.path.join(src, f, '8'))]
    for f in dir_list:
        print(f)
    #    m = f.split('-')[0]
        f = os.path.join(src, f)
    #    n, sl = select_slice(f)
    #    sitk.WriteImage(sl, os.path.join(dst, '{0}_{1}.tif'.format(m, n)), True)
        flatten_fix(f)
