import numpy as np
import random
from PIL import Image
import cv2
from tqdm import tqdm


class StyleAugment:
    """
    Class for applying FDA (Fouried Domain Adaptation) style augmentation to images.

    Args:
        n_images_per_style (int): Number of images per style.
        L (float): Parameter for computing the size of the style patch.
        size (tuple): Desired size of the images after preprocessing.
        b (int): Size of the style patch.

    Methods:
        preprocess(self, x): Preprocesses the input image.
        deprocess(self, x, size): Deprocesses the input image.
        add_style(self, loader, multiple_styles=False, name=None): Adds styles to the style pool.
        _extract_style(self, img_np): Extracts the style from an image.
        compute_size(self, amp_shift): Computes the size of the style patch.
        apply_style(self, image): Applies a random style to the input image.
        _apply_style(self, img): Applies a specific style to the input image.
        test(self, images_np, images_target_np=None, size=None): Tests the style augmentation on input images.
    """
    def __init__(self, n_images_per_style=10, L=0.1, size=(1024, 512), b=None):
        self.styles = []
        self.styles_names = []
        self.n_images_per_style = n_images_per_style
        self.L = L
        self.size = size
        self.sizes = None
        self.cv2 = False
        self.b = b

    def preprocess(self, x):
        """
        Preprocesses the input image.

        Args:
            x: Input image.

        Returns:
            np.ndarray: Preprocessed image.
        """
        if isinstance(x, np.ndarray):
            x = cv2.resize(x, self.size, interpolation=cv2.INTER_CUBIC)
            self.cv2 = True
        else:
            x = x.resize(self.size, Image.BICUBIC)
        x = np.asarray(x, np.float32)
        x = x[:, :, ::-1]
        x = x.transpose((2, 0, 1))
        return x.copy()

    def deprocess(self, x, size):
        """
        Deprocesses the input image.

        Args:
            x: Input image.
            size: Desired size of the output image.

        Returns:
            PIL.Image.Image: Deprocessed image.
        """
        if self.cv2:
            x = cv2.resize(np.uint8(x).transpose((1, 2, 0))[:, :, ::-1], size, interpolation=cv2.INTER_CUBIC)
        else:
            x = Image.fromarray(np.uint8(x).transpose((1, 2, 0))[:, :, ::-1])
            x = x.resize(size, Image.BICUBIC)
        return x

    def add_style(self, loader, multiple_styles=False, name=None):
        """
        Adds styles to the style pool.

        Args:
            loader: Data loader for loading images.
            multiple_styles (bool): Whether to add multiple styles per image.
            name (str): Name of the style.

        Returns:
            None
        """
        if self.n_images_per_style < 0:
            return

        if name is not None:
            self.styles_names.append([name] * self.n_images_per_style if multiple_styles else [name])

        loader.return_unprocessed_image = True
        n = 0
        styles = []
        print(len(loader))
        for sample in tqdm(loader, total=min(len(loader), self.n_images_per_style)):

            image = self.preprocess(sample)

            if n >= self.n_images_per_style:
                break
            styles.append(self._extract_style(image))
            n += 1

        if self.n_images_per_style > 1:
            if multiple_styles:
                self.styles += styles
            else:
                styles = np.stack(styles, axis=0)
                style = np.mean(styles, axis=0)
                self.styles.append(style)
        elif self.n_images_per_style == 1:
            self.styles += styles

        loader.return_unprocessed_image = False

    def _extract_style(self, img_np):
        """
        Extracts the style from an image.

        Args:
            img_np: Input image.

        Returns:
            np.ndarray: Extracted style.
        """
        fft_np = np.fft.fft2(img_np, axes=(-2, -1))
        amp = np.abs(fft_np)
        amp_shift = np.fft.fftshift(amp, axes=(-2, -1))
        if self.sizes is None:
            self.sizes = self.compute_size(amp_shift)
        h1, h2, w1, w2 = self.sizes
        style = amp_shift[:, h1:h2, w1:w2]
        return style

    def compute_size(self, amp_shift):
        """
        Computes the size of the style patch.

        Args:
            amp_shift: Shifted amplitude spectrum.

        Returns:
            tuple: Size parameters (h1, h2, w1, w2) of the style patch.
        """
        _, h, w = amp_shift.shape
        b = (np.floor(np.amin((h, w)) * self.L)).astype(int) if self.b is None else self.b
        c_h = np.floor(h / 2.0).astype(int)
        c_w = np.floor(w / 2.0).astype(int)
        h1 = c_h - b
        h2 = c_h + b + 1
        w1 = c_w - b
        w2 = c_w + b + 1
        return h1, h2, w1, w2

    def apply_style(self, image):
        return self._apply_style(image)

    def _apply_style(self, img):
        """
        Applies a random style to the input image.

        Args:
            image: Input image.

        Returns:
            PIL.Image.Image: Image with applied style.
        """
        if self.n_images_per_style < 0:
            return img

        if len(self.styles) > 0:
            n = random.randint(0, len(self.styles) - 1)
            style = self.styles[n]
        else:
            style = self.styles[0]

        if isinstance(img, np.ndarray):
            H, W = img.shape[0:2]
        else:
            W, H = img.size
        img_np = self.preprocess(img)

        fft_np = np.fft.fft2(img_np, axes=(-2, -1))
        amp, pha = np.abs(fft_np), np.angle(fft_np)
        amp_shift = np.fft.fftshift(amp, axes=(-2, -1))
        h1, h2, w1, w2 = self.sizes
        amp_shift[:, h1:h2, w1:w2] = style
        amp_ = np.fft.ifftshift(amp_shift, axes=(-2, -1))

        fft_ = amp_ * np.exp(1j * pha)
        img_np_ = np.fft.ifft2(fft_, axes=(-2, -1))
        img_np_ = np.real(img_np_)
        img_np__ = np.clip(np.round(img_np_), 0., 255.)

        img_with_style = self.deprocess(img_np__, (W, H))

        return img_with_style
    
    
    def test(self, images_np, images_target_np=None, size=None):
        """
        Tests the style augmentation on input images.

        Args:
            images_np: Input images.
            images_target_np: Target images (optional).
            size: Desired size of the images (optional).

        Returns:
            None
        """
        Image.fromarray(np.uint8(images_np.transpose((1, 2, 0)))[:, :, ::-1]).show()
        fft_np = np.fft.fft2(images_np, axes=(-2, -1))
        amp = np.abs(fft_np)
        amp_shift = np.fft.fftshift(amp, axes=(-2, -1))
        h1, h2, w1, w2 = self.sizes
        style = amp_shift[:, h1:h2, w1:w2]

        fft_np_ = np.fft.fft2(images_np if images_target_np is None else images_target_np, axes=(-2, -1))
        amp_, pha_ = np.abs(fft_np_), np.angle(fft_np_)
        amp_shift_ = np.fft.fftshift(amp_, axes=(-2, -1))
        h1, h2, w1, w2 = self.sizes
        amp_shift_[:, h1:h2, w1:w2] = style
        amp__ = np.fft.ifftshift(amp_shift_, axes=(-2, -1))

        fft_ = amp__ * np.exp(1j * pha_)
        img_np_ = np.fft.ifft2(fft_, axes=(-2, -1))
        img_np_ = np.real(img_np_)
        img_np__ = np.clip(np.round(img_np_), 0., 255.)
        Image.fromarray(np.uint8(images_target_np.transpose((1, 2, 0)))[:, :, ::-1]).show()
        Image.fromarray(np.uint8(img_np__).transpose((1, 2, 0))[:, :, ::-1]).show()
