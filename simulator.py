# -*- coding: utf-8 -*-

"""

"""
import matplotlib.pyplot as plt
import numpy as np
import copy
import pickle
from tqdm.auto import trange
from scipy.ndimage import binary_dilation
from mpl_toolkits.axes_grid1 import make_axes_locatable

import lenstronomy.Util.data_util as data_util
import lenstronomy.Util.util as util
import lenstronomy.Plots.plot_util as plot_util
from lenstronomy.Util.param_util import phi_q2_ellipticity
from lenstronomy.SimulationAPI.sim_api import SimAPI
from lenstronomy.Workflow.fitting_sequence import FittingSequence
from lenstronomy.Plots.model_plot import ModelPlot
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver

# plot settings
import seaborn as sns

# to change tex to Times New Roman in mpl
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['mathtext.rm'] = 'serif'
plt.rcParams['mathtext.it'] = 'serif:italic'
plt.rcParams['mathtext.bf'] = 'serif:bold'
plt.rcParams['mathtext.fontset'] = 'custom'


def set_fontscale(font_scale=1):
    sns.set(style='ticks', context=None,
            font='Times New Roman',
            rc={#"text.usetex": True,
                #"font.family": 'serif',
                #"font.serif": 'Times New Roman',
                #"mathtext.rm": 'serif',
                #"mathtext.it": 'serif:italic',
                #"mathtext.bf": 'serif:bold',
                #"mathtext.fontset": 'custom',
                "xtick.direction": "in",
                "ytick.direction": "in",
                "axes.linewidth": 0.5*font_scale,
                "axes.labelsize": 9*font_scale,
                "font.size": 9*font_scale,
                "axes.titlesize": 9*font_scale,
                "legend.fontsize": 8*font_scale,
                "xtick.labelsize": 8*font_scale,
                "ytick.labelsize": 8*font_scale,
               })


set_fontscale(2.)
palette = sns.color_palette('muted', 8)
palette.as_hex()


class Simulator(object):

    def __init__(self, lens_specifications, filter_specifications,
                 observing_scenarios, psfs,
                 magnitude_distributions, use_pemd=False,
                 source_galaxy_indices=[]
                 ):
        self.num_lenses = lens_specifications['num_lenses']
        self._with_quasar = lens_specifications['with_quasar']

        self.filter_specifications = filter_specifications
        self.observing_scenarios = observing_scenarios

        self.simulation_psfs = psfs['simulation']
        self.modeling_psfs = psfs['modeling']

        if 'psf_uncertainty_level' in psfs:
            self._psf_uncertainty_level = psfs['psf_uncertainty_level']
        else:
            self._psf_uncertainty_level = 0.

        self.lens_magnitude_distributions = magnitude_distributions['lens']
        self.source_magnitude_distributions = magnitude_distributions['source']
        if self._with_quasar:
            self.quasar_magnitude_distributions = magnitude_distributions[
                                                                    'quasar']

        self.num_pixels = self.filter_specifications['num_pixel']
        self.pixel_scales = self.filter_specifications['pixel_scale']

        self.num_filters = self.filter_specifications['num_filter']
        self.num_scenarios = len(self.observing_scenarios)

        self._kwargs_model = {
            'lens_model_list': ['PEMD' if use_pemd else 'EPL', 'SHEAR'],
            'lens_light_model_list': ['SERSIC_ELLIPSE'],
            'source_light_model_list': ['SHAPELETS'],
            'point_source_model_list': ['SOURCE_POSITION'] if
            self._with_quasar else []
        }
        self._kwargs_model_smooth_source = {
            'lens_model_list': ['PEMD' if use_pemd else 'EPL', 'SHEAR'],
            'lens_light_model_list': ['SERSIC_ELLIPSE'],
            'source_light_model_list': ['SERSIC_ELLIPSE'],
            'point_source_model_list': ['SOURCE_POSITION'] if
            self._with_quasar else []
        }

        self._shapelet_coeffs = np.load(
            'source_galaxy_shapelet_coefficients_nmax50.npz')['arr_0']

        self._kwargs_lenses = []
        self._source_positions = []
        self._lens_ellipticities = []
        self._source_ellipticities = []
        if self._with_quasar:
            self._image_positions = []
        else:
            self._image_positions = None

        if source_galaxy_indices == []:
            source_galaxy_indices = np.random.randint(0,
                                  len(self._shapelet_coeffs), self.num_lenses)
        self._source_galaxy_shapelet_coeffs = self._shapelet_coeffs[
                                                        source_galaxy_indices]

        for j in range(self.num_lenses):
            q = np.random.uniform(0.7, 0.9)
            phi = np.random.uniform(-90, 90)
            self._lens_ellipticities.append([q, phi])

            e1, e2 = phi_q2_ellipticity(phi*np.pi/180, q)

            theta_E = np.random.uniform(1.2, 1.6)
            self._kwargs_lenses.append([
                {'theta_E': theta_E,
                 'gamma': np.random.uniform(1.9, 2.1),
                 'e1': e1,
                 'e2': e2,
                 'center_x': 0, 'center_y': 0},
                {'gamma1': np.random.uniform(-0.08, 0.08),
                 'gamma2': np.random.uniform(-0.08, 0.08),
                 'ra_0': 0,
                 'dec_0': 0}
            ])

            r = np.random.uniform(0.05, 0.35) * theta_E
            phi = np.random.uniform(-np.pi, np.pi)
            self._source_positions.append([r * np.cos(phi), r * np.sin(phi)])
            self._source_ellipticities.append([
                np.random.uniform(-0.3, 0.3), np.random.uniform(-0.3, 0.3)
            ])

            if self._with_quasar:
                self._image_positions.append(
                    self._get_point_image_positions(
                        self._kwargs_lenses[-1],
                        self._source_positions[-1]
                ))

        self._weighted_exposure_time_maps = \
            self._get_weighted_exposure_time_maps()
        self.sim_apis = self._get_sim_apis(self._kwargs_model)
        self.sim_apis_smooth_source = self._get_sim_apis(
            self._kwargs_model_smooth_source)
        self.image_sims = self._get_image_sims(self.sim_apis)

        self._kwargs_light = self._get_kwargs_light()
        self.simulated_data = self._simulate_data()

        self._walker_ratio = 8

    def _get_point_image_positions(self, kwargs_lens,
                                   source_position):
        lens_model = LensModel(self._kwargs_model['lens_model_list'])
        lens_equation_solver = LensEquationSolver(lens_model)

        x_image, y_image = lens_equation_solver.image_position_from_source(
                    kwargs_lens=kwargs_lens, sourcePos_x=source_position[0],
                    sourcePos_y=source_position[1], min_distance=0.01,
                    search_window=5,
                    precision_limit=10 ** (-10), num_iter_max=100)

        return x_image, y_image

    def _get_weighted_exposure_time_maps(self):
        weighted_exposure_time_maps = []

        for j in range(self.num_lenses):
            weighted_exposure_time_maps_scenarios = []
            for n in range(self.num_scenarios):
                weighted_exposure_time_maps_filters = []
                for i in range(self.num_filters):
                    simulate_cosmic_ray = False
                    if 'simulate_cosmic_ray' in self.observing_scenarios[n]:
                        if not self.observing_scenarios[n][
                                                'simulate_cosmic_ray'][i]:
                            simulate_cosmic_ray = False
                        else:
                            simulate_cosmic_ray = True
                            if self.observing_scenarios[n][
                                    'simulate_cosmic_ray'][i]:
                                cosmic_ray_count_rate = 2.4e-3
                            else:
                                cosmic_ray_count_rate = \
                                        self.observing_scenarios[n][
                                            'simulate_cosmic_ray'][i]

                    if simulate_cosmic_ray:
                        weighted_exposure_time_maps_filters.append(
                            self._make_weighted_exposure_time_map(
                                self.observing_scenarios[n]['exposure_time'][i],
                                self.num_pixels[i],
                                self.pixel_scales[i],
                                self.observing_scenarios[n]['num_exposure'][i],
                                cosmic_ray_count_rate
                            )
                        )
                    else:
                        weighted_exposure_time_maps_filters.append(
                            np.ones((self.num_pixels[i], self.num_pixels[i])) *
                            self.observing_scenarios[n]['exposure_time'][i])

                weighted_exposure_time_maps_scenarios.append(
                    weighted_exposure_time_maps_filters)
            weighted_exposure_time_maps.append(
                weighted_exposure_time_maps_scenarios)

        return weighted_exposure_time_maps

    @property
    def walker_ratio(self):
        if hasattr(self, '_walker_ratio'):
            return self._walker_ratio
        else:
            self._walker_ratio = 8
            return self._walker_ratio

    def set_walker_ratio(self, ratio):
        self._walker_ratio = ratio

    def plot_simualated_data(self, vmax=None, vmin=None, figsize=None):
        nrows = self.num_lenses
        ncols = self.num_scenarios * self.num_filters
        fig, axes = plt.subplots(nrows=nrows,
                                 ncols=ncols,
                                 figsize=figsize if figsize else
                                 (max(nrows * 3, 10), max(ncols * 5, 6))
                                 )
        if nrows == 1 and ncols == 1:
            axes = [[axes]]
        elif len(axes.shape) == 1:
            axes = [axes]

        if vmax is None:
            vmax = [2] * self.num_filters

        if vmin is None:
            vmin = [-4] * self.num_filters

        for j in range(self.num_lenses):
            for n in range(self.num_scenarios):
                for i in range(self.num_filters):
                    axes[j][n*self.num_filters+i].matshow(
                        np.log10(self.simulated_data[j][n][i]),
                        cmap='cubehelix', origin='lower',
                        vmin=vmin[i],
                        vmax=vmax[i]
                    )
                    axes[j][n * self.num_filters + i].set_xticks([])
                    axes[j][n * self.num_filters + i].set_yticks([])
                    axes[j][n * self.num_filters + i].set_aspect('equal')

                    if j == 0:
                        axes[j][n * self.num_filters + i].set_title(
                            'Scenario: {}, filter: {}'.format(n+1, i+1))

                    if n == 0 and i == 0:
                        axes[j][n * self.num_filters + i].set_ylabel('Lens: '
                                                             '{}'.format(j+1))

        fig.tight_layout()
        return fig

    def plot_exposure_maps(self, figsize=None):
        nrows = self.num_lenses
        ncols = self.num_scenarios * self.num_filters
        fig, axes = plt.subplots(nrows=nrows,
                                 ncols=ncols,
                                 figsize=figsize if figsize else
                                    (max(nrows*3, 10), max(ncols*5, 6))
                                 )

        if nrows == 1 and ncols == 1:
            axes = [[axes]]
        elif len(axes.shape) == 1:
            axes = [axes]

        for j in range(self.num_lenses):
            for n in range(self.num_scenarios):
                for i in range(self.num_filters):
                    im = axes[j][n*self.num_filters+i].matshow(
                        self._weighted_exposure_time_maps[j][n][i] *
                        self.observing_scenarios[n]['num_exposure'][i],
                        cmap='viridis', origin='lower', vmin=0
                    )
                    divider = make_axes_locatable(axes[j][
                                                      n*self.num_filters+i])
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    plt.colorbar(im, cax=cax, label='(seconds)')
                    axes[j][n * self.num_filters + i].set_xticks([])
                    axes[j][n * self.num_filters + i].set_yticks([])
                    axes[j][n * self.num_filters + i].set_aspect('equal')

                    if j == 0:
                        axes[j][n * self.num_filters + i].set_title(
                            'Scenario: {}, filter: {}'.format(n+1, i+1))

                    if n == 0 and i == 0:
                        axes[j][n * self.num_filters + i].set_ylabel('Lens: '
                                                             '{}'.format(j+1))

        fig.tight_layout()
        return fig

    def _simulate_data(self):
        simulated_data_lenses = []

        for j in range(self.num_lenses):
            simulated_data_scenarios = []

            for n in range(self.num_scenarios):
                simulated_data_filters = []

                for i in range(self.num_filters):
                    kwargs_lens_light, kwargs_source, \
                    kwargs_ps = self._kwargs_light[j][n][i]
                    simulated_image = self.image_sims[j][n][i].image(
                        self._kwargs_lenses[j],
                        kwargs_source, kwargs_lens_light, kwargs_ps,
                        source_add=True, lens_light_add=True,
                        point_source_add=True if self._with_quasar else False
                        )

                    simulated_image[simulated_image < 0 ] = 1e-10

                    simulated_image += self.sim_apis[j][n][i].noise_for_model(
                       model=simulated_image)

                    simulated_data_filters.append(simulated_image)

                simulated_data_scenarios.append(simulated_data_filters)
            simulated_data_lenses.append(simulated_data_scenarios)

        return simulated_data_lenses

    def _get_image_sims(self, sim_apis):
        image_sims = []
        for j in range(self.num_lenses):
            image_sims_scenarios = []
            for n in range(self.num_scenarios):
                image_sim_filters = []
                for i in range(self.num_filters):
                    kwargs_numerics = {
                        'point_source_supersampling_factor':
                            self.filter_specifications[
                                'simulation_psf_supersampling_resolution'][i],
                        'supersampling_factor': 3
                    }

                    image_sim_filters.append(
                        sim_apis[j][n][i].image_model_class(kwargs_numerics)
                    )

                image_sims_scenarios.append(image_sim_filters)
            image_sims.append(image_sims_scenarios)

        return image_sims

    def _get_sim_apis(self, kwargs_model):
        sim_apis = []

        for j in range(self.num_lenses):
            sim_api_scenarios = []
            for n in range(self.num_scenarios):
                sim_api_filters = []
                kwargs_observation = self._get_filter_kwargs(j, n)
                for i in range(self.num_filters):
                    sim_api_filters.append(SimAPI(numpix=self.num_pixels[i],
                                       kwargs_single_band=kwargs_observation[i],
                                       kwargs_model=kwargs_model))

                sim_api_scenarios.append(sim_api_filters)
            sim_apis.append(sim_api_scenarios)

        return sim_apis

    def _make_weighted_exposure_time_map(self, exposure_time, num_pixel,
                                         pixel_scale, num_exposure,
                                         cosmic_ray_count_rate=2.4e-3):
        exposure_time_map = np.ones((num_pixel, num_pixel)) * exposure_time

        cosmic_ray_weight_map = 0.
        for i in range(num_exposure):
            cosmic_ray_count = cosmic_ray_count_rate * (num_pixel *
                                            pixel_scale)**2 * exposure_time

            cosmic_ray_weight_map += self._create_cr_hitmap(num_pixel,
                                                            pixel_scale,
                                                            cosmic_ray_count
                                                            )

        exposure_time_map *= cosmic_ray_weight_map / num_exposure

        exposure_time_map[exposure_time_map == 0.] = 1e-10

        return exposure_time_map

    def _get_filter_kwargs(self, n_lens, scenario_index):
        filter_kwargs = []
        for i in range(self.num_filters):
            exposure_time = self._weighted_exposure_time_maps[n_lens][
                scenario_index][i]
            filter_kwargs.append(
                {
                    'read_noise': self.filter_specifications['read_noise'][i],
                    'ccd_gain': self.filter_specifications['ccd_gain'][i],
                    'sky_brightness': self.filter_specifications[
                        'sky_brightness'][i],
                    'magnitude_zero_point':
                        self.filter_specifications[
                            'magnitude_zero_point'][i],
                    'exposure_time': exposure_time,
                    'num_exposures': self.observing_scenarios[
                        scenario_index]['num_exposure'][i],
                    'seeing': self.filter_specifications['seeing'][i],
                    'pixel_scale': self.filter_specifications[
                        'pixel_scale'][i],
                    'psf_type': 'PIXEL',
                    'kernel_point_source': self.simulation_psfs[i],
                    'point_source_supersampling_factor': self.filter_specifications[
                            'simulation_psf_supersampling_resolution'][i]
                })

        return filter_kwargs

    def _get_kwargs_light(self):
        kwargs_light_lenses = []

        for j in range(self.num_lenses):
            kwargs_light_scenarios = []

            lens_magnitudes = self.lens_magnitude_distributions()
            source_magnitudes = self.source_magnitude_distributions()
            if self._with_quasar:
                ps_magnitudes = self.quasar_magnitude_distributions()

            source_R_sersic = np.random.uniform(0.1, 0.2)

            for n in range(self.num_scenarios):
                kwargs_light = []

                for i in range(self.num_filters):
                    q, phi = self._lens_ellipticities[j]
                    e1, e2 = phi_q2_ellipticity(phi*np.pi/180., q)

                    kwargs_lens_light_mag = [{
                        'magnitude': lens_magnitudes[i],
                        'R_sersic': 1.,
                        'n_sersic': 4,
                        'e1': e1, 'e2': e2,
                        'center_x': 0, 'center_y': 0
                    }]

                    kwargs_source_light_mag = [{
                        'magnitude': source_magnitudes[i],
                        'R_sersic': source_R_sersic,
                        'n_sersic': 1,
                        'e1': self._source_ellipticities[j][0],
                        'e2': self._source_ellipticities[j][0],
                        'center_x': self._source_positions[j][0],
                        'center_y': self._source_positions[j][1]
                    }]

                    kwargs_ps_mag = [{
                        'ra_source': self._source_positions[j][0],
                        'dec_source': self._source_positions[j][1],
                        'magnitude': ps_magnitudes[i]
                    }] if self._with_quasar else []

                    kwargs_lens_light, kwargs_source_smooth, kwargs_ps = \
                        self.sim_apis_smooth_source[j][n][
                            i].magnitude2amplitude(
                            kwargs_lens_light_mag, kwargs_source_light_mag,
                            kwargs_ps_mag)

                    smooth_light_model = LightModel(['SERSIC_ELLIPSE'])
                    shapelet_light_model = LightModel(['SHAPELETS'])

                    x, y = util.make_grid(200, 0.01)
                    smooth_flux = np.sum(smooth_light_model.surface_brightness(
                                            x, y, kwargs_source_smooth))

                    kwargs_source = [{
                        'n_max': self.filter_specifications[
                            'simulation_shapelet_n_max'][i],
                        'beta': source_R_sersic,
                        'amp': self._source_galaxy_shapelet_coeffs[j],
                        'center_x': self._source_positions[j][0],
                        'center_y': self._source_positions[j][1]
                    }]

                    shapelet_flux = np.sum(
                        shapelet_light_model.surface_brightness(
                                            x, y, kwargs_source))

                    kwargs_source[0]['amp'] *= smooth_flux / shapelet_flux

                    kwargs_light.append([kwargs_lens_light, kwargs_source,
                                         kwargs_ps])

                kwargs_light_scenarios.append(kwargs_light)

            kwargs_light_lenses.append(kwargs_light_scenarios)

        return kwargs_light_lenses

    def _get_kwargs_data(self, n_lens, n_scenario):
        kwargs_data_list = []

        for i in range(self.num_filters):
            kwargs_data_list.append({
                'image_data': self.simulated_data[n_lens][n_scenario][i],
                'background_rms': self.sim_apis[n_lens][n_scenario][
                                                    i].background_noise,
                'noise_map': None,
                'exposure_time': (self._weighted_exposure_time_maps[n_lens][
                                     n_scenario][i] *
                                    self.observing_scenarios[n_scenario][
                                        'num_exposure'][i]),
                'ra_at_xy_0': -(self.num_pixels[i] - 1)/2. * self.pixel_scales[i],
                'dec_at_xy_0': -(self.num_pixels[i] - 1)/2. * self.pixel_scales[i],
                'transform_pix2angle': np.array([[self.pixel_scales[i], 0],
                                                [0, self.pixel_scales[i]]
                                                ])
            })

        return kwargs_data_list

    def _get_kwargs_psf(self, n_lens, n_scenario):
        kwargs_psf_list = []
        for i in range(self.num_filters):
            if self._psf_uncertainty_level > 0.:
                max_noise = np.max(self.modeling_psfs[i]) * self._psf_uncertainty_level
                exposure_time = np.max(self.modeling_psfs[i]) / max_noise**2
                # F*t = (N*t)^2
                psf_uncertainty = np.sqrt(self.modeling_psfs[i] *
                                          exposure_time) / exposure_time
            kwargs_psf_list.append({
                'psf_type': "PIXEL",
                'kernel_point_source': self.modeling_psfs[i],
                'kernel_point_source_init': self.modeling_psfs[i],
                'psf_error_map': psf_uncertainty,
                'point_source_supersampling_factor': self.filter_specifications[
                            'modeling_psf_supersampling_resolution'][i]
            })

        return kwargs_psf_list

    def _get_kwargs_params(self, n_lens, n_scenario):
        # initial guess of non-linear parameters, starting from the truth
        # for fast convergence of the MCMC
        kwargs_lens_init = self._kwargs_lenses[n_lens]

        kwargs_lens_light_init = [
            self._kwargs_light[n_lens][n_scenario][i][0][0] for i in range(
                self.num_filters)
        ]
        kwargs_source_init = [
            self._kwargs_light[n_lens][n_scenario][i][1][0] for i in range(
                self.num_filters)
        ]
        for i in range(self.num_filters):
            kwargs_source_init[i]['n_max'] = self.filter_specifications[
                'modeling_shapelet_n_max'][i]

        kwargs_ps_init = [
            self._kwargs_light[n_lens][n_scenario][0][2][0]
        ] if self._with_quasar else []

        if self._with_quasar:
            num_image = len(self._image_positions[n_lens][0])
            kwargs_ps_init[0]['ra_source'] = kwargs_source_init[0]['center_x']
            kwargs_ps_init[0]['dec_source'] = kwargs_source_init[0]['center_y']
            # kwargs_ps_init[0]['ra_image'] = self._image_positions[n_lens][0]
            # kwargs_ps_init[0]['dec_image'] = self._image_positions[n_lens][1]

        # initial spread in parameter estimation
        kwargs_lens_sigma = [
            {'theta_E': 0.01, 'e1': 0.01, 'e2': 0.01, 'gamma': .02,
             'center_x': 0.05, 'center_y': 0.05},
            {'gamma1': 0.01, 'gamma2': 0.01}]
        kwargs_lens_light_sigma = [
            {'R_sersic': 0.05, 'n_sersic': 0.1, 'e1': 0.01, 'e2': 0.01,
             'center_x': .01, 'center_y': 0.01} for _ in range(
                self.num_filters)]
        kwargs_source_sigma = [
            {'beta': 0.01,
             #'n_sersic': .05, 'e1': 0.05, 'e2': 0.05,
             'center_x': 0.05, 'center_y': 0.05} for _ in range(
                self.num_filters)]
        kwargs_ps_sigma = [{#'ra_image': 5e-5*np.ones(num_image),
                            #'dec_image': 5e-5*np.ones(num_image),
                            'ra_source': 5e-5,
                            'dec_source': 5e-5
                            }] if self._with_quasar else []

        # hard bound lower limit in parameter space
        kwargs_lower_lens = [
            {'theta_E': 0, 'e1': -0.5, 'e2': -0.5, 'gamma': 1.5,
             'center_x': -10., 'center_y': -10},
            {'gamma1': -0.5, 'gamma2': -0.5}]
        kwargs_lower_source = [
            {'beta': 0.001,
             #'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5,
             'center_x': -10, 'center_y': -10} for _ in range(
                self.num_filters)]
        kwargs_lower_lens_light = [
            {'R_sersic': 0.001, 'n_sersic': 0.5, 'e1': -0.5, 'e2': -0.5,
             'center_x': -10, 'center_y': -10} for _ in range(
                self.num_filters)]
        kwargs_lower_ps = [{#'ra_image': -1.5*np.ones(num_image),
                            #'dec_image': -1.5*np.ones(num_image),
                            'ra_source': -1.5,
                            'dec_source': -1.5
                            }] if self._with_quasar else []

        # hard bound upper limit in parameter space
        kwargs_upper_lens = [
            {'theta_E': 10, 'e1': 0.5, 'e2': 0.5, 'gamma': 2.5,
             'center_x': 10., 'center_y': 10},
            {'gamma1': 0.5, 'gamma2': 0.5}]
        kwargs_upper_source = [
            {'beta': 10,
             #'n_sersic': 5., 'e1': 0.5, 'e2': 0.5,
             'center_x': 10, 'center_y': 10} for _ in range(self.num_filters)]
        kwargs_upper_lens_light = [
            {'R_sersic': 10, 'n_sersic': 5., 'e1': 0.5, 'e2': 0.5,
             'center_x': 10, 'center_y': 10} for _ in range(self.num_filters)]
        kwargs_upper_ps = [{#'ra_image': 1.5*np.ones(num_image),
                            #'dec_image': 1.5*np.ones(num_image)
                            'ra_source': 1.5,
                            'dec_source': 1.5
                            }] if self._with_quasar else []

        # keeping parameters fixed
        kwargs_lens_fixed = [{}, {'ra_0': 0, 'dec_0': 0}]
        kwargs_source_fixed = [{'n_max': self.filter_specifications[
            'modeling_shapelet_n_max'][i]} for i in range(
                                        self.num_filters)]
        kwargs_lens_light_fixed = [{} for _ in range(self.num_filters)]
        kwargs_ps_fixed = [{}] if self._with_quasar else []

        lens_params = [kwargs_lens_init, kwargs_lens_sigma, kwargs_lens_fixed,
                       kwargs_lower_lens, kwargs_upper_lens]
        source_params = [kwargs_source_init, kwargs_source_sigma,
                         kwargs_source_fixed, kwargs_lower_source,
                         kwargs_upper_source]
        lens_light_params = [kwargs_lens_light_init, kwargs_lens_light_sigma,
                             kwargs_lens_light_fixed, kwargs_lower_lens_light,
                             kwargs_upper_lens_light]
        ps_params = [kwargs_ps_init, kwargs_ps_sigma, kwargs_ps_fixed,
                     kwargs_lower_ps, kwargs_upper_ps]

        kwargs_params = {'lens_model': lens_params,
                         'source_model': source_params,
                         'lens_light_model': lens_light_params,
                         'point_source_model': ps_params}

        return kwargs_params

    def _get_multi_band_list(self, n_lens, n_scenario):
        kwargs_data_list = self._get_kwargs_data(n_lens, n_scenario)
        kwargs_psf_list = self._get_kwargs_psf(n_lens, n_scenario)

        multi_band_list = []
        for i in range(self.num_filters):
            psf_supersampling_factor = self.filter_specifications[
                                'simulation_psf_supersampling_resolution'][i]
            kwargs_numerics = {'supersampling_factor': 3,
                               'supersampling_convolution': True if
                               psf_supersampling_factor > 1 else False,
                               'supersampling_kernel_size': 5,
                               'point_source_supersampling_factor':
                                   psf_supersampling_factor,
                               'compute_mode': 'adaptive',
                               }



            image_band = [kwargs_data_list[i], kwargs_psf_list[i],
                          kwargs_numerics]

            multi_band_list.append(image_band)

        return multi_band_list

    def _get_kwargs_constraints(self, n_lens, n_scenario):
        kwargs_constraints = {
            'joint_lens_with_light': [[0, 0, ['center_x',
                                              'center_y'
                                              ]]] if not
            self._with_quasar else [],
            'joint_lens_light_with_lens_light': [[0, i, ['center_x',
                                                         'center_y',
                                                         'e1', 'e2',
                                                         'n_sersic'
                                                         ]] for i
                                                 in range(1,
                                                          self.num_filters)],
            'joint_source_with_source': [[0, i, ['center_x',
                                                 'center_y',
                                                 'beta'
                                                 ]] for i
                                         in range(1, self.num_filters)],
            'joint_source_with_point_source': [[0, 0]] if self._with_quasar
                                                                    else [],
            # 'num_point_source_list': None,
            # 'solver_type': 'None'
        }

        if self._with_quasar:
            num_images = len(self._image_positions[n_lens][0])
            # kwargs_constraints['solver_type'] = 'PROFILE_SHEAR' if \
            #     num_images == 4 else 'CENTER'
            # kwargs_constraints['num_point_source_list'] = [num_images]

        return kwargs_constraints

    def _get_kwargs_likelihood(self, n_lens, n_scenario):
        total_exposure_times = np.array(self.observing_scenarios[n_scenario][
                                            'exposure_time']) \
                               * np.array(self.observing_scenarios[n_scenario][
                                              'num_exposure'])
        bands_compute = []
        for time in total_exposure_times:
            bands_compute.append(True if time > 0 else False)

        mask_list = []

        for i in range(self.num_filters):
            if 'simulate_cosmic_ray' in self.observing_scenarios[n_scenario]:
                if self.observing_scenarios[n_scenario]['simulate_cosmic_ray'][i]:
                    weighted_exposure_time_map = \
                        self._weighted_exposure_time_maps[n_lens][n_scenario][i]
                    mask = np.ones_like(weighted_exposure_time_map)
                    mask[weighted_exposure_time_map <= 1e-10] = 0.
                    mask_list.append(mask)
                else:
                    mask_list.append(None)
            else:
                mask_list.append(None)

        # for galaxy-galxy lenses
        kwargs_likelihood = {
            'force_no_add_image': False,
            'source_marg': False,
            # 'point_source_likelihood': True,
            # 'position_uncertainty': 0.00004,
            # 'check_solver': False,
            # 'solver_tolerance': 0.001,
            'check_positive_flux': True,
            'check_bounds': True,
            'bands_compute': bands_compute,
            'image_likelihood_mask_list': mask_list
        }

        return kwargs_likelihood

    def _fit_one_model(self, n_lens, n_scenario, num_threads=1, n_run=500):
        multi_band_list = self._get_multi_band_list(n_lens, n_scenario)

        kwargs_data_joint = {'multi_band_list': multi_band_list,
                             'multi_band_type': 'multi-linear'}

        kwargs_params = self._get_kwargs_params(n_lens, n_scenario)
        kwargs_model = self._get_kwargs_model()

        kwargs_constraints = self._get_kwargs_constraints(n_lens, n_scenario)
        kwargs_likelihood = self._get_kwargs_likelihood(n_lens, n_scenario)

        fitting_seq = FittingSequence(kwargs_data_joint, kwargs_model,
                                      kwargs_constraints, kwargs_likelihood,
                                      kwargs_params)

        fitting_kwargs_list = [
            ['MCMC',
             {'n_burn': 0, 'n_run': n_run, 'walkerRatio': 8,
              'sigma_scale': 1e-4, 'progress': True,
              'threadCount': num_threads}]
        ]

        chain_list = fitting_seq.fit_sequence(fitting_kwargs_list)
        kwargs_result = fitting_seq.best_fit()

        return [[kwargs_data_joint, kwargs_model,
                 kwargs_constraints, kwargs_likelihood, kwargs_params],
                [chain_list, kwargs_result]]

    def _extend_chain(self, n_lens, n_scenario, run_id, num_threads=1,
                      n_run=500, save_directory='./temp/'):

        save_file = save_directory + '{}_lens_{}_scenario_{' \
                                     '}.pickle'.format(run_id, n_lens,
                                                        n_scenario)

        with open(save_file, 'rb') as f:
            [input, output] = pickle.load(f)

        [kwargs_data_joint, kwargs_model,
         kwargs_constraints, kwargs_likelihood, kwargs_params] = input

        chain_list = output[0]
        samples_mcmc = chain_list[0][1]

        n_params = samples_mcmc.shape[1]
        n_walkers = self.walker_ratio * n_params
        n_step = int(samples_mcmc.shape[0] / n_walkers)

        print('N_step: {}, N_walkers: {}, N_params: {}'.format(n_step,
                                                               n_walkers,
                                                               n_params))

        chain = np.empty((n_walkers, n_step, n_params))

        for i in np.arange(n_params):
            samples = samples_mcmc[:, i].T
            chain[:, :, i] = samples.reshape((n_step, n_walkers)).T

        fitting_seq = FittingSequence(kwargs_data_joint, kwargs_model,
                                      kwargs_constraints, kwargs_likelihood,
                                      kwargs_params)

        fitting_kwargs_list = [
            ['MCMC',
             {'n_burn': 0, 'n_run': n_run, 'walkerRatio': 8,
              'init_samples': chain[:, -1, :],
              #'sigma_scale': 3,
              'progress': True,
              'threadCount': num_threads}]
        ]

        new_chain_list = fitting_seq.fit_sequence(fitting_kwargs_list)
        kwargs_result = fitting_seq.best_fit()

        chain_list.append(new_chain_list[0])

        return [[kwargs_data_joint, kwargs_model,
                 kwargs_constraints, kwargs_likelihood, kwargs_params],
                [new_chain_list, kwargs_result]]


    def _get_kwargs_model(self):
        kwargs_model = copy.deepcopy(self._kwargs_model)
        kwargs_model['lens_light_model_list'] = [self._kwargs_model[
                                    'lens_light_model_list'][0] for _ in range(
            self.num_filters)]
        kwargs_model['source_light_model_list'] = [self._kwargs_model[
                                'source_light_model_list'][0] for _ in range(
            self.num_filters)]
        kwargs_model['index_lens_light_model_list'] = [[i] for i in range(
            self.num_filters)]
        kwargs_model['index_source_light_model_list'] = [[i] for i in range(
            self.num_filters)]

        if self._with_quasar:
            kwargs_model['point_source_model_list'] = ['SOURCE_POSITION']

        return kwargs_model

    def fit_models(self, run_id='', num_threads=1, n_run=500,
                   save_directory='./temp/'):
        for j in range(self.num_lenses):
            for n in range(self.num_scenarios):
                print('Running lens: {}/{}, scenario: {}/{}'.format(
                    j+1, self.num_lenses, n+1, self.num_scenarios
                ))
                model_fit = self._fit_one_model(
                    j, n,
                    num_threads=num_threads,
                    n_run=n_run
                    )

                save_file = save_directory + '{}_lens_{}_scenario_{' \
                                             '}.pickle'.format(run_id, j, n)

                with open(save_file, 'wb') as f:
                    pickle.dump(model_fit, f)

    def extend_chains(self, num_lenses, num_scenarios,
                      run_id='', extend_id='', num_threads=1, n_run=500,
                      save_directory='./temp/'):
        for j in range(num_lenses):
            for n in range(num_scenarios):
                print('Running lens: {}/{}, scenario: {}/{}'.format(
                    j+1, self.num_lenses, n+1, self.num_scenarios
                ))

                model_fit = self._extend_chain(
                    j, n, run_id,
                    num_threads=num_threads,
                    n_run=n_run, save_directory=save_directory
                    )

                save_file = save_directory + '{}{}_lens_{}_scenario_{' \
                                             '}.pickle'.format(run_id,
                                                               extend_id, j, n)

                with open(save_file, 'wb') as f:
                    pickle.dump(model_fit, f)

    @classmethod
    def plot_lens_models(self, run_id, num_lens, num_scenario, num_filters=1,
                   save_directory='./temp/'):
        save_file = save_directory + '{}_lens_{}_scenario_{' \
                                     '}.pickle'.format(run_id, num_lens,
                                                       num_scenario)

        with open(save_file, 'rb') as f:
            [input, output] = pickle.load(f)

        multi_band_list = input[0]['multi_band_list']
        kwargs_model = input[1]
        kwargs_likelihood = input[3]
        kwargs_result = output[1]

        lens_plot = ModelPlot(multi_band_list, kwargs_model,
                              kwargs_result,
                              arrow_size=0.02,  # cmap_string=cmap,
                              likelihood_mask_list=kwargs_likelihood[
                                                'image_likelihood_mask_list'],
                              multi_band_type='multi-linear',
                              cmap_string='cubehelix',
                              # , source_marg=True, linear_prior=[1e5, 1e5, 1e5]
                              )

        fig, axes = plt.subplots(num_filters, 3,
                               figsize=(num_filters*8, 10),
                               sharex=False, sharey=False)

        if num_filters == 1:
            axes = [axes]

        for i in range(num_filters):
            lens_plot.data_plot(ax=axes[i][0], band_index=i,
                                v_max=2, v_min=-4,
                                text='Filter {}'.format(i+1))
            lens_plot.model_plot(ax=axes[i][1], band_index=i,
                                 v_max=2, v_min=-4)
            lens_plot.normalized_residual_plot(ax=axes[i][2], band_index=i,
                                               v_max=5, v_min=-5, cmap='RdBu')

        return fig

    def plot_mcmc_trace(self, run_id, n_lens, n_scenario,
                        save_directory='./temp/'):
        save_file = save_directory + '{}_lens_{}_scenario_{' \
                                     '}.pickle'.format(run_id, n_lens,
                                                       n_scenario)

        with open(save_file, 'rb') as f:
            [_, output] = pickle.load(f)

        chain_list = output[0]
        samples_mcmc = chain_list[-1][1]
        param_mcmc = chain_list[-1][2]

        n_params = samples_mcmc.shape[1]
        n_walkers = self.walker_ratio * n_params
        n_step = int(samples_mcmc.shape[0] / n_walkers)

        print('N_step: {}, N_walkers: {}, N_params: {}'.format(n_step,
                                                               n_walkers,
                                                               n_params))

        chain = np.empty((n_walkers, n_step, n_params))

        for i in np.arange(n_params):
            samples = samples_mcmc[:, i].T
            chain[:, :, i] = samples.reshape((n_step, n_walkers)).T

        mean_pos = np.zeros((n_params, n_step))
        median_pos = np.zeros((n_params, n_step))
        std_pos = np.zeros((n_params, n_step))
        q16_pos = np.zeros((n_params, n_step))
        q84_pos = np.zeros((n_params, n_step))

        for i in np.arange(n_params):
            for j in np.arange(n_step):
                mean_pos[i][j] = np.mean(chain[:, j, i])
                median_pos[i][j] = np.median(chain[:, j, i])
                std_pos[i][j] = np.std(chain[:, j, i])
                q16_pos[i][j] = np.percentile(chain[:, j, i], 16.)
                q84_pos[i][j] = np.percentile(chain[:, j, i], 84.)

        fig, ax = plt.subplots(n_params, sharex=True, figsize=(8, 6))

        burnin = -1
        last = n_step

        medians = []

        # param_values = [median_pos[0][last - 1],
        #                 (q84_pos[0][last - 1] - q16_pos[0][last - 1]) / 2,
        #                 median_pos[1][last - 1],
        #                 (q84_pos[1][last - 1] - q16_pos[1][last - 1]) / 2]

        for i in range(n_params):
            print(param_mcmc[i],
                  '{:.4f} ± {:.4f}'.format(median_pos[i][last - 1], (
                              q84_pos[i][last - 1] - q16_pos[i][
                          last - 1]) / 2))

            ax[i].plot(median_pos[i][:last], c='g')
            ax[i].axhline(np.median(median_pos[i][burnin:last]), c='r',
                          lw=1)
            ax[i].fill_between(np.arange(last), q84_pos[i][:last],
                               q16_pos[i][:last], alpha=0.4)
            ax[i].set_ylabel(param_mcmc[i], fontsize=10)
            ax[i].set_xlim(0, last)

            medians.append(np.median(median_pos[i][burnin:last]))

        fig.set_size_inches((12., 2 * len(param_mcmc)))

        return fig

    @staticmethod
    def _create_cr_hitmap(num_pix, pixel_scale, cosmic_ray_count):
        map = np.ones((num_pix, num_pix))
        image_size = num_pix * pixel_scale

        for i in range(10):
            n_cr = int(np.random.normal(loc=cosmic_ray_count,
                                        scale=np.sqrt(cosmic_ray_count)
                                     ))

            if n_cr > 0:
                break

        if n_cr < 1:
            n_cr = 0
            
        for i in range(n_cr):
            x = np.random.randint(0, num_pix)
            y = np.random.randint(0, num_pix)

            threshold = 1.
            while True:
                map[x, y] = 0
                direction = np.random.randint(0, 4)

                if direction == 0:
                    x += 1
                elif direction == 1:
                    y += 1
                elif direction == 2:
                    x -= 1
                else:
                    y -= 1

                if x < 0:
                    x = 0
                if x >= num_pix:
                    x = num_pix-1
                if y < 0:
                    y = 0
                if y >= num_pix:
                    y = num_pix-1

                toss = np.random.uniform(0, 1.)

                if toss > threshold:
                    break

                threshold -= (0.05 * (pixel_scale/0.04)**4)
                    
        return 1 - binary_dilation(1 - map)

    def compare_model_parameter(self, parameter_name, run_id, num_lenses,
                                num_scenarios, save_directory='./temp/',
                                clip_chain=-10,
                                ):

        if parameter_name == 'gamma':
            param_index = 1
            param_latex = r'$\gamma$'
        elif parameter_name == 'theta_E':
            param_index = 0
            param_latex = r'$\theta_{\rm E}$'
        else:
            raise ValueError('Parameter {} not supported!'.format(
                                                            parameter_name))

        parameter_posteriors = []
        parameter_truths = []

        for n_lens in range(num_lenses):
            parameter_posterior_scenarios = []
            for n_scenario in range(num_scenarios):
                save_file = save_directory + '{}_lens_{}_scenario_{' \
                                         '}.pickle'.format(run_id, n_lens,
                                                           n_scenario)

                with open(save_file, 'rb') as f:
                    [input, output] = pickle.load(f)

                chain_list = output[0]
                samples_mcmc = chain_list[-1][1]
                #param_mcmc = chain_list[-1][2]

                if n_scenario == 0:
                    kwargs_params = input[4]
                    parameter_truths.append(kwargs_params['lens_model'][0][0][
                        parameter_name])

                n_params = samples_mcmc.shape[1]
                n_walkers = self.walker_ratio * n_params
                n_step = int(samples_mcmc.shape[0] / n_walkers)

                # print('N_step: {}, N_walkers: {}, N_params: {}'.format(n_step,
                #                                                        n_walkers,
                #                                                        n_params))

                chain = np.empty((n_walkers, n_step, n_params))

                for i in np.arange(n_params):
                    samples = samples_mcmc[:, i].T
                    chain[:, :, i] = samples.reshape((n_step, n_walkers)).T

                low, mid, hi = np.percentile(chain[:, clip_chain:n_step,
                                             param_index], q=[16, 50, 84])

                parameter_posterior_scenarios.append([low, mid, hi])

            parameter_posteriors.append(parameter_posterior_scenarios)

        fig, axes = plt.subplots(ncols=2, figsize=(20, 6))

        for i in range(num_lenses):
            axes[0].plot(i+1, parameter_truths[i], marker='x', c='k',
                         label='Truth' if i==0 else None
                         )
            for j in range(num_scenarios):
                axes[0].errorbar(i+1+(j+1)*0.1,
                                 parameter_posteriors[i][j][1],
                                 yerr=(parameter_posteriors[i][j][2]
                                       - parameter_posteriors[i][j][ 0])/2.,
                                 marker='o',
                                 label='scenario {}'.format(j+1) if i == 0
                                        else None,
                                 color=palette[j]
                                 )

                axes[1].bar(i+1+j*0.1, (parameter_posteriors[i][j][2]
                                           - parameter_posteriors[i][j][0])/2.,
                            width=0.1,
                            label='scenario {}'.format(j+1) if i == 0
                                        else None,
                            color=palette[j]
                            )

        axes[0].set_ylabel(param_latex)
        axes[1].set_ylabel('Uncertainty')

        axes[0].set_xlabel('Lens index')
        axes[1].set_xlabel('Lens index')

        axes[0].legend()
        axes[1].legend()

        return fig, (parameter_truths, parameter_posteriors)
