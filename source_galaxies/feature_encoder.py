# -*- coding: utf-8 -*-
"""
This module contains the class to extract features (shapelet coefficients) from
galaxy images.
"""

import numpy as np
from scipy.optimize import dual_annealing

from lenstronomy.Util.analysis_util import half_light_radius
from lenstronomy.LightModel.Profiles.shapelets import ShapeletSet
import lenstronomy.Util.util as util
import lenstronomy.Util.image_util as image_util


class FeatureEncoder(object):
    """
    This class extracts shapelet coefficients as features of a galaxy.
    """

    def __init__(self, pixel_size, image_width):
        """
        Initiate the class with provided values for class variables.
        :param pixel_size: size of pixel in images in arcsecond
        :type pixel_size: `float`
        :param image_width: width and height of images in pixel number
        :type image_width: `int`
        :return: `None`
        :rtype: `None`
        """
        self.pixel_size = pixel_size
        self.image_width = image_width

        # make a coordinate grid
        self.x_, self.y_ = util.make_grid(numPix=self.image_width,
                                          deltapix=self.pixel_size)

        self.shapelet_set = ShapeletSet()

        # pixel coordinates at the center of the image
        self.image_center_xpix = (self.image_width - 1.) / 2.
        self.image_center_ypix = (self.image_width - 1.) / 2.

        # arcsec coordinates at (0, 0) pixel
        self.pix_00_x_arcsec = -self.image_center_xpix * self.pixel_size
        self.pix_00_y_arcsec = -self.image_center_ypix * self.pixel_size

    def pix2arcsec(self, xpix, ypix):
        """
        Convert pixel coordinates to arcsec coordinates.
        :return:
        :rtype:
        """
        x_arcsec = self.pix_00_x_arcsec + (xpix*self.pixel_size)
        y_arcsec = self.pix_00_y_arcsec + (ypix*self.pixel_size)

        return x_arcsec, y_arcsec

    def arcsec2pix(self, x_arcsec, y_arcsec):
        """
        Convert arcsec coordinates to pixel coordinates.
        :param x_arcsec:
        :type x_arcsec:
        :param y_arcsec:
        :type y_arcsec:
        :return:
        :rtype:
        """
        xpix = (-self.pix_00_x_arcsec + x_arcsec) / self.pixel_size
        ypix = (-self.pix_00_y_arcsec + y_arcsec) / self.pixel_size

        return xpix, ypix

    def get_offset_from_center(self, image, mask_radius=0.5):
        """
        Get offset of the brightest pixel assumed as the galaxy center from the
        center of the image in arcsecond.
        :param image:
        :type image:
        :param mask_radius: boundry for the max offset, in arcsec
        :type mask_radius: `float`
        :return: offset of the galaxy center from the image center
        :rtype: `(float, float)`
        """
        # create a mask of (mask_radius/pixel_size) pixels
        x, y = np.meshgrid(np.linspace(-(self.image_width-1)/2.,
                                       (self.image_width-1)/2.,
                                       self.image_width),
                           np.linspace(-(self.image_width - 1) / 2.,
                                       (self.image_width - 1) / 2.,
                                       self.image_width),
                           )
        r = np.sqrt(x*x + y*y)

        inside = (r <= 0.5 / 0.05)
        outside = (r > 0.5 / 0.05)

        r[inside] = 1.
        r[outside] = 0.

        center = np.where(image*r == np.max(image*r))
        # here 1 is x and 0 is column, because (row, column) is
        # reverse of (x, y)
        #center = np.array([center[1], center[0]]).squeeze()
        offset = self.pix2arcsec(center[1][0], center[0][0])

        return offset

    def get_scale_radius(self, image):
        """
        Get the radius in arcsecond that contains half of the total flux in the
        image.
        :param image: 2d image
        :type image: `ndarray`
        :return: 1/5th of the half-light radius within the image
        :rtype: `float`
        """
        x_grid, y_grid = self.x_, self.y_
        image_1d = util.image2array(image)
        scale_radius = half_light_radius(image_1d, x_grid, y_grid)

        return scale_radius / 5.

    def get_shapelet_coefficients(self, image, n_max, beta, rotate=0.,
                                  offset=(0., 0.)):
        """
        Get shapelet coefficients from decomposition of given image.
        :param image:
        :type image:
        :param n_max:
        :type n_max:
        :param beta:
        :type beta:
        :param rotate:
        :type rotate:
        :param offset:
        :rtype offset:
        :return:
        :rtype:
        """

        x = self.x_ * np.cos(rotate * np.pi / 180.) - self.y_ * np.sin(
            rotate * np.pi / 180)
        y = self.x_ * np.sin(rotate * np.pi / 180.) + self.y_ * np.cos(
            rotate * np.pi / 180)

        x_off_rot = offset[0] * np.cos(rotate * np.pi / 180.) - offset[1] * \
                    np.sin(rotate * np.pi / 180)
        y_off_rot = offset[0] * np.sin(rotate * np.pi / 180.) + offset[1] * \
                    np.cos(rotate * np.pi / 180)
        image_1d = util.image2array(image)

        coefficients = self.shapelet_set.decomposition(image_1d, x, y,
                                                       n_max, beta,
                                                       self.pixel_size,
                                                       center_x=x_off_rot,
                                                       center_y=y_off_rot)

        return coefficients

    def get_chi2(self, image, noise_map, shapelet_coefficients,
                 n_max, beta, offset=(0., 0.)):
        """
        Compute the chi^2 of the shapelet reconstruction of an image.
        :param image:
        :type image:
        :param noise_map:
        :type noise_map:
        :param shapelet_coefficients:
        :type shapelet_coefficients:
        :param n_max:
        :type n_max:
        :param beta:
        :type beta:
        :param offset:
        :type offset:
        :return:
        :rtype:
        """
        reconstructed_image = self.decode_image(shapelet_coefficients,
                                                n_max, beta, offset=offset
                                                )

        return np.sum((image - reconstructed_image)**2/2./noise_map**2)

    def get_best_beta(self, image, noise_map, n_max, offset=(0., 0.),
                      min=0.025, max=1.025, step_size=0.025):
        """

        :param image:
        :type image:
        :param noise_map:
        :type noise_map:
        :param n_max:
        :type n_max:
        :param offset:
        :type offset:
        :param min:
        :type min:
        :param max:
        :type max:
        :param step_size:
        :type step_size:
        :return:
        :rtype:
        """
        betas = np.arange(min, max, step_size)
        chi2s = []

        for beta in betas:
            coefficients = self.get_shapelet_coefficients(
                image, n_max, beta, offset=offset
            )

            chi2s.append(self.get_chi2(image, noise_map, coefficients,
                                       n_max, beta, offset=offset))

        return betas[np.argmin(chi2s)]

    def encode_image(self, image, noise_map, n_max, beta=None, rotate=0.,
                     offset=None):
        """
        Encode the image for the given `n_max`. First find offset to the
        center of the galaxy, then find the best shapelet :math:`\beta`
        parameter.
        :param image:
        :type image:
        :param noise_map:
        :type noise_map:
        :param n_max:
        :type n_max:
        :param beta:
        :type beta:
        :param rotate:
        :type rotate:
        :param offset:
        :type offset:
        :return:
        :rtype:
        """
        if offset is None:
            offset = self.get_offset_from_center(image)

        if beta is None:
            beta = self.get_best_beta(image, noise_map, n_max, offset=offset)

        return self.get_shapelet_coefficients(image, n_max, beta,
                                              rotate=rotate, offset=offset)

    def decode_image(self, shapelet_coefficients, n_max, beta=None,
                     offset=(0., 0.)):
        """

        :param shapelet_coefficients:
        :type shapelet_coefficients:
        :param n_max:
        :type n_max:
        :param beta:
        :type beta:
        :param offset:
        :rtype offset:
        :return:
        :rtype:
        """
        if beta is None:
            beta = self.pixel_size * self.image_width / 4.

        image_reconstructed = self.shapelet_set.function(
            self.x_, self.y_,
            shapelet_coefficients, n_max,
            beta, center_x=offset[0], center_y=offset[1]
        )

        return util.array2image(image_reconstructed)

    def subtract_background(self, image, box_size=10,
                            with_background_rms=False):
        """
        Subtract background noise level from image.
        :param image:
        :type image:
        :param box_size:
        :type box_size:
        :param with_background_rms: if `True`, return the background rms too
        :type with_background_rms: `bool`
        :return:
        :rtype:
        """
        noise_cutouts = []

        bs0p5 = int(box_size/2)
        bs1p5 = bs0p5*3
        width_m_bs1p5 = self.image_width - bs1p5
        width_m_bs = self.image_width - box_size
        corners = [(0, 0), (0, bs0p5),
                   (bs0p5, 0), (bs0p5, bs0p5),
                   (width_m_bs1p5, width_m_bs1p5), (width_m_bs1p5, width_m_bs),
                   (width_m_bs, width_m_bs1p5), (width_m_bs, width_m_bs),
                   (0, width_m_bs1p5), (bs0p5, width_m_bs1p5),
                   (bs0p5, width_m_bs), (0, width_m_bs),
                   (width_m_bs1p5, 0), (width_m_bs1p5, bs0p5),
                   (width_m_bs, bs0p5), (width_m_bs, 0)
                   ]

        for c in corners:
            noise_cutouts.append(image[c[0]:c[0] + 10, c[1]:c[1] + 10])

        background = np.nanmedian([np.nanmean(a) for a in noise_cutouts][:-2])

        # print(np.mean(noise_cutouts, axis=(1, 2)))

        if with_background_rms:
            background_rms = np.nanmedian([np.std(a) for a in
                                           noise_cutouts[:-2]])

            return image - background, background_rms
        else:
            return image - background