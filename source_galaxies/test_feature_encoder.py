# -*- coding: utf-8 -*-
"""
Tests for config module.
"""

import numpy as np
import numpy.testing as npt
import lenstronomy.Util.util as util
from lenstronomy.LightModel.Profiles.shapelets import ShapeletSet

from .feature_encoder import FeatureEncoder


class TestFeatureEncoder(object):

    def setup_class(self):
        self.feature_encoder = FeatureEncoder(0.05, 100)
        self.shapelet_set = ShapeletSet()

    @classmethod
    def teardown_class(cls):
        pass

    def test_init(self):
        """
        Test `__init__` method.
        :return:
        :rtype:
        """
        assert self.feature_encoder.pixel_size == 0.05
        assert self.feature_encoder.image_width == 100

        assert (self.feature_encoder.image_center_xpix,
                self.feature_encoder.image_center_ypix) == (49.5, 49.5)
        assert (self.feature_encoder.pix_00_x_arcsec,
                self.feature_encoder.pix_00_x_arcsec) == (-2.475, -2.475)

    def test_pix2arcsec(self):
        """

        :return:
        :rtype:
        """
        assert self.feature_encoder.pix2arcsec(49.5, 49.5) == (0., 0.)
        assert self.feature_encoder.pix2arcsec(0, 0) == (-2.475, -2.475)
        assert self.feature_encoder.pix2arcsec(99, 99) == (2.475, 2.475)

    def test_arcsec2pix(self):
        """

        :return:
        :rtype:
        """
        assert self.feature_encoder.arcsec2pix(0., 0.) == (49.5, 49.5)
        assert self.feature_encoder.arcsec2pix(-2.475, -2.475) == (0, 0)
        assert self.feature_encoder.arcsec2pix(2.475, 2.475) == (99, 99)

    def test_get_offset_from_center(self):
        """
        Test `get_offset_from_center` method.
        :return:
        :rtype:
        """
        image = np.zeros((100, 100))

        image[49, 49] = 1.
        npt.assert_almost_equal(
            self.feature_encoder.get_offset_from_center(image),
            np.array([-0.025, -0.025]), 3)

        image[51, 49] = 2.
        npt.assert_almost_equal(
            self.feature_encoder.get_offset_from_center(image),
            np.array([-0.025, 0.075]), 3)

        image[49, 51] = 3.
        npt.assert_almost_equal(
            self.feature_encoder.get_offset_from_center(image),
            np.array([0.075, -0.025]), 3)

    def test_get_scale_radius(self):
        """
        Test `get_scale_radius` method by checking a 100x100 array of 1's.
        :return:
        :rtype:
        """
        r = 100. * np.sqrt(1/2./np.pi) * self.feature_encoder.pixel_size / 5.
        npt.assert_almost_equal(self.feature_encoder.get_scale_radius(
            np.ones((100, 100))), r, 2)

    def test_get_shapelet_coefficients(self):
        """
        Test `get_shapelet_coefficients` method. Create an image from given
        shapelet coefficients first, then check if the retrieved coefficients
        from `get_shapelet_coefficients` match.
        :return:
        :rtype:
        """
        test_coefficients = np.array([
            366.60963256,   -4.07288105,   84.86054911,  135.06647642,
            -74.3030201 ,   75.28430753,   -0.93696042,   25.15496810,
            -30.86780729,   12.55242697,   87.38230830,  -53.25264982,
             54.00527839,   -6.95030545,    9.35410895
        ])

        beta = 0.5
        n_max = 4

        image_reconstructed = self.shapelet_set.function(
            self.feature_encoder.x_, self.feature_encoder.y_,
            test_coefficients, n_max,
            beta, center_x=0, center_y=0
        )

        coefficients = self.feature_encoder.get_shapelet_coefficients(
            util.array2image(image_reconstructed),
            n_max, beta
        )

        npt.assert_almost_equal(test_coefficients, coefficients, 4)

    def test_get_chi2(self):
        """
        Test `get_chi2` method.
        :return:
        :rtype:
        """
        test_coefficients = np.array([
            366.60963256, -4.07288105, 84.86054911, 135.06647642,
            -74.3030201, 75.28430753, -0.93696042, 25.15496810,
            -30.86780729, 12.55242697, 87.38230830, -53.25264982,
            54.00527839, -6.95030545, 9.35410895
        ])

        beta = 0.5
        n_max = 4

        image = util.array2image(self.shapelet_set.function(
            self.feature_encoder.x_, self.feature_encoder.y_,
            test_coefficients, n_max,
            beta, center_x=0, center_y=0
        ))

        chi2 = self.feature_encoder.get_chi2(image, np.ones_like(image)*0.01,
                                             test_coefficients, n_max,
                                             beta
                                             )

        npt.assert_almost_equal(chi2, 0., 5)

    def test_get_best_beta(self):
        """
        Test `get_best_beta` method.
        :return:
        :rtype:
        """
        test_coefficients = np.array([
            366.60963256, -4.07288105, 84.86054911, 135.06647642,
            -74.3030201, 75.28430753, -0.93696042, 25.15496810,
            -30.86780729, 12.55242697, 87.38230830, -53.25264982,
            54.00527839, -6.95030545, 9.35410895
        ])

        beta = 0.27
        n_max = 4

        image = util.array2image(self.shapelet_set.function(
            self.feature_encoder.x_, self.feature_encoder.y_,
            test_coefficients, n_max,
            beta, center_x=0, center_y=0
        ))

        best_beta = self.feature_encoder.get_best_beta(
            image, np.ones_like(image)*0.01, n_max
        )

        assert np.abs(beta - best_beta) <= self.feature_encoder.pixel_size/2.

    def test_encode_image(self):
        """
        Test `encode_image` method.
        :return:
        :rtype:
        """
        test_coefficients = np.array([
            366.60963256, -4.07288105, 84.86054911, 135.06647642,
            -74.3030201, 75.28430753, -0.93696042, 25.15496810,
            -30.86780729, 12.55242697, 87.38230830, -53.25264982,
            54.00527839, -6.95030545, 9.35410895
        ])

        beta = 0.275
        n_max = 4

        image = util.array2image(self.shapelet_set.function(
            self.feature_encoder.x_, self.feature_encoder.y_,
            test_coefficients, n_max,
            beta, center_x=0, center_y=0
        ))

        encoded = self.feature_encoder.encode_image(image,
                                                    np.ones_like(image)*0.01,
                                                    n_max, offset=(0, 0))

        npt.assert_almost_equal(test_coefficients, encoded, 4)

    def test_decode_image(self):
        """
        Test `decode_image` method.
        :return:
        :rtype:
        """
        test_coefficients = np.array([
            366.60963256, -4.07288105, 84.86054911, 135.06647642,
            -74.3030201, 75.28430753, -0.93696042, 25.15496810,
            -30.86780729, 12.55242697, 87.38230830, -53.25264982,
            54.00527839, -6.95030545, 9.35410895
        ])

        beta = 0.5
        n_max = 4

        image_reconstructed = util.array2image(self.shapelet_set.function(
            self.feature_encoder.x_, self.feature_encoder.y_,
            test_coefficients, n_max,
            beta, center_x=0, center_y=0
        ))

        image_decoded = self.feature_encoder.decode_image(
            test_coefficients, n_max, beta
        )

        npt.assert_almost_equal(image_reconstructed, image_decoded, 5)

    def test_subtract_background(self):
        """
        Test `subtract_background` method.
        :return:
        :rtype:
        """
        image = np.random.normal(loc=1., scale=0.1, size=(100, 100))

        npt.assert_almost_equal(np.mean(
            self.feature_encoder.subtract_background(image)), 0., 2)

        _, background_rms = self.feature_encoder.subtract_background(image,
                                                    with_background_rms=True)
        npt.assert_almost_equal(background_rms, 0.1, 2)


if __name__ == '__main__':
    pytest.main()