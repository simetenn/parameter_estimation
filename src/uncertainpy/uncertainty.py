from __future__ import absolute_import, division, print_function, unicode_literals

import os
import platform
import numpy as np

from .core.uncertainty_calculations import UncertaintyCalculations
from .plotting.plot_uncertainty import PlotUncertainty
from .utils.logger import get_logger, add_file_handler
from .data import Data
from .core.base import ParameterBase


class UncertaintyQuantification(ParameterBase):
    """
    Perform an uncertainty quantification and sensitivity analysis of a model
    and features of the model.

    It implements both quasi-Monte Carlo methods and polynomial chaos expansions
    using either point collocation or the pseudo-spectral method. Both of the
    polynomial chaos expansion methods have support for the rosenblatt
    transformation to handle dependent input parameters.

    Parameters
    ----------
    model : {None, Model or Model subclass instance, model function}
        Model to perform uncertainty quantification on. For requirements see
        Model.run.
        Default is None.
    parameters : {None, Parameters instance, list of Parameter instances, list with [[name, value, distribution], ...]}
        Either None, a Parameters instance or a list of the parameters that should be created.
        The two lists are similar to the arguments sent to Parameters.
        Default is None.
    features : {None, Features or Features subclass instance, list of feature functions}, optional
        Features to calculate from the model result.
        If None, no features are calculated.
        If list of feature functions, all will be calculated.
        Default is None.
    uncertainty_calculations : UncertaintyCalculations or UncertaintyCalculations subclass instance, optional
        An UncertaintyCalculations class or subclass that implements (custom)
        uncertainty quantification and sensitivity analysis methods.
    create_PCE_custom : callable, optional
        A custom method for calculating the polynomial chaos approximation.
        For the requirements of the function see
        ``UncertaintyCalculations.create_PCE_custom``. Overwrites existing
        ``create_PCE_custom`` method.
        Default is None.
    custom_uncertainty_quantification : callable, optional
        A custom method for calculating uncertainties.
        For the requirements of the function see
        ``UncertaintyCalculations.custom_uncertainty_quantification``.
        Overwrites existing ``custom_uncertainty_quantification`` method.
        Default is None.
    CPUs : {int, None, "max"}, optional
        The number of CPUs to use when calculating the model and features.
        If None, no multiprocessing is used.
        If "max", the maximum number of CPUs on the computer
        (multiprocess.cpu_count()) is used.
        Default is "max".
    logger_level : {"info", "debug", "warning", "error", "critical", None}, optional
        Set the threshold for the logging level. Logging messages less severe
        than this level is ignored. If None, no logging to file is performed
        Default logger level is "info".
    logger_filename : str
        Name of the logfile. If None, no logging to file is performed. Default is
        "uncertainpy.log".
    backend : {"auto", "hdf5", "exdir"}, optional
        The fileformat used to save and load data to/from file. "auto" assumes the
        filenames ends with either ".h5" for HDF5 files or ".exdir" for Exdir files.
        If unknown fileextension defaults to saving data as HDF5 files. "hdf5" saves
        and loads files from HDF5 files. "exdir" saves and loads files from
        Exdir files. Default is "auto".

    Attributes
    ----------
    model : Model or Model subclass
        The model to perform uncertainty quantification on.
    parameters : Parameters
        The uncertain parameters.
    features : Features or Features subclass
        The features of the model to perform uncertainty quantification on.
    uncertainty_calculations : UncertaintyCalculations or UncertaintyCalculations subclass
        UncertaintyCalculations object responsible for performing the uncertainty
        quantification calculations.
    data : Data
        A data object that contains the results from the uncertainty quantification.
        Contains all model and feature evaluations, as well as all calculated
        statistical metrics.

    Raises
    ------
    ValueError
        If unsupported backend is chosen.

    See Also
    --------
    uncertainpy.features
    uncertainpy.Parameter
    uncertainpy.Parameters
    uncertainpy.models
    uncertainpy.core.UncertaintyCalculations
    uncertainpy.core.UncertaintyCalculations.create_PCE_custom : Requirements for create_PCE_custom
    uncertainpy.models.Model.run : Requirements for the model run function.
    """
    def __init__(self,
                 model,
                 parameters,
                 features=None,
                 uncertainty_calculations=None,
                 create_PCE_custom=None,
                 custom_uncertainty_quantification=None,
                 CPUs="max",
                 logger_level="info",
                 logger_filename="uncertainpy.log",
                 backend="auto"):


        if backend not in ["auto", "hdf5", "exdir"]:
            raise ValueError("backend {} not supported. Supported backends are: auto, hdf5, and exdir".format(backend))


        logger = get_logger(self)

        if platform.system().lower() == "windows":
            logger.info("On Windows machines everything in your script must be "
                        "inside of an if __name__ == '__main__': block in order "
                        "for multiprocess to work." )

        if uncertainty_calculations is None:
            self._uncertainty_calculations = UncertaintyCalculations(
                model=model,
                parameters=parameters,
                features=features,
                create_PCE_custom=create_PCE_custom,
                custom_uncertainty_quantification=custom_uncertainty_quantification,
                CPUs=CPUs,
                logger_level=logger_level,
            )
        else:
            self._uncertainty_calculations = uncertainty_calculations

        super(UncertaintyQuantification, self).__init__(parameters=parameters,
                                                        model=model,
                                                        features=features,
                                                        logger_level=logger_level)


        self.data = None
        self.backend = backend

        self.plotting = PlotUncertainty(folder=None,
                                        logger_level=logger_level)

        add_file_handler(filename=logger_filename)


    @ParameterBase.features.setter
    def features(self, new_features):
        ParameterBase.features.fset(self, new_features)

        self.uncertainty_calculations.features = self.features


    @ParameterBase.model.setter
    def model(self, new_model):
        ParameterBase.model.fset(self, new_model)

        self.uncertainty_calculations.model = self.model


    @ParameterBase.parameters.setter
    def parameters(self, new_parameters):
        ParameterBase.parameters.fset(self, new_parameters)

        self.uncertainty_calculations.parameters = self.parameters


    @property
    def uncertainty_calculations(self):
        """
        The class for performing the calculations for the uncertainty
        quantification and sensitivity analysis.

        Parameters
        ----------
        new_uncertainty_calculations : UncertaintyCalculations or UncertaintyCalculations subclass instance
            New UncertaintyCalculations object responsible for performing the uncertainty
            quantification calculations.

        Returns
        -------
        uncertainty_calculations : UncertaintyCalculations or UncertaintyCalculations subclass instance
            UncertaintyCalculations object responsible for performing the uncertainty
            quantification calculations.

        See Also
        --------
        uncertainpy.core.UncertaintyCalculations
        """
        return self._uncertainty_calculations


    @uncertainty_calculations.setter
    def uncertainty_calculations(self, new_uncertainty_calculations):
        self._uncertainty_calculations = new_uncertainty_calculations

        self.uncertainty_calculations.features = self.features
        self.uncertainty_calculations.model = self.model


    # TODO add features_to_run as argument to this function
    def quantify(self,
                 method="pc",
                 pc_method="collocation",
                 rosenblatt="auto",
                 uncertain_parameters=None,
                 polynomial_order=4,
                 nr_collocation_nodes=None,
                 quadrature_order=None,
                 nr_pc_mc_samples=10**4,
                 nr_mc_samples=10**4,
                 allow_incomplete=True,
                 seed=None,
                 single=False,
                 plot="condensed_first",
                 figure_folder="figures",
                 figureformat=".png",
                 save=True,
                 save_samples=False,
                 data_folder="data",
                 filename=None,
                 **custom_kwargs):
        """
        Perform an uncertainty quantification and sensitivity analysis
        using polynomial chaos expansions or quasi-Monte Carlo methods.

        Parameters
        ----------
        method : {"pc", "mc", "custom"}, optional
            The method to use when performing the uncertainty quantification and
            sensitivity analysis.
            "pc" is polynomial chaos method, "mc" is the quasi-Monte Carlo
            method and "custom" are custom uncertainty quantification methods.
            Default is "pc".
        pc_method : {"collocation", "spectral", "custom"}, optional
            The method to use when creating the polynomial chaos approximation,
            if the polynomial chaos method is chosen. "collocation" is the
            point collocation method "spectral" is pseudo-spectral projection,
            and "custom" is the custom polynomial method.
            Default is "collocation".
        rosenblatt : {"auto", bool}, optional
            If the Rosenblatt transformation should be used. The Rosenblatt
            transformation must be used if the uncertain parameters have
            dependent variables. If "auto" the Rosenblatt transformation is used
            if there are dependent parameters, and it is not used of the
            parameters have independent distributions. Default is "auto".
        uncertain_parameters : {None, str, list}, optional
            The uncertain parameter(s) to use when performing the uncertainty
            quantification. If None, all uncertain parameters are used.
            Default is None.
        polynomial_order : int, optional
            The polynomial order of the polynomial approximation.
            Default is 4.
        nr_collocation_nodes : {int, None}, optional
            The number of collocation nodes to choose, if polynomial chaos with
            point collocation is used. If None,
            `nr_collocation_nodes` = 2* number of expansion factors + 2.
            Default is None.
        quadrature_order : {int, None}, optional
            The order of the Leja quadrature method, if polynomial chaos with
            pseudo-spectral projection is used. If None,
            ``quadrature_order = polynomial_order + 2``.
            Default is None.
        nr_pc_mc_samples : int, optional
            Number of samples for the Monte Carlo sampling of the polynomial
            chaos approximation, if the polynomial chaos method is chosen.
            Default is 10**4.
        nr_mc_samples : int, optional
            Number of samples for the quasi-Monte Carlo sampling, if the quasi-Monte
            Carlo method is chosen. `nr_mc_samples` is used for the uncertainty
            quantification and ``(nr_mc_samples/2)*(nr_uncertain_parameters + 2)``
            samples is used for the sensitivity analysis. Default `nr_mc_samples`
            is 10**4.
        allow_incomplete : bool, optional
            If the polynomial approximation should be performed for features or
            models with incomplete evaluations.
            Default is True.
        seed : int, optional
            Set a random seed. If None, no seed is set.
            Default is None.
        single : bool
            If an uncertainty quantification should be performed with only one
            uncertain parameter at the time. Requires that the values of
            each parameter is set. Default is False.
        plot : {"condensed_first", "condensed_total", "condensed_no_sensitivity", "all", "evaluations", None}, optional
            Type of plots to be created.
            "condensed_first" is a subset of the most important plots and
            only plots each result once, and contains plots of the first order
            Sobol indices. "condensed_total" is similar, but with the
            total order Sobol indices, and "condensed_no_sensitivity" is the
            same without any Sobol indices plotted. "all" creates every plot.
            "evaluations" plots the model and feature evaluations. None plots
            nothing.
            Default is "condensed_first".
        figure_folder : str, optional
            Name of the folder where to save all figures.
            Default is "figures".
        figureformat : str
            The figure format to save the plots in. Supports all formats in
            matplolib.
            Default is ".png".
        save : bool, optional
            If the data should be saved. Default is True.
        data_folder : str, optional
            Name of the folder where to save the data. Default is "data".
        filename : {None, str}, optional
            Name of the data file. If None the model name is used.
            Default is None.
        **custom_kwargs
            Any number of arguments for either the custom polynomial chaos method,
            ``create_PCE_custom``, or the custom uncertainty quantification,
            ``custom_uncertainty_quantification``.

        Returns
        -------
        data : Data, dict containing data objects
            A data object that contains the results from the uncertainty quantification.
            Contains all model and feature evaluations, as well as all calculated
            statistical metrics. If `single` = True, then returns a dictionary
            that contains the data objects for each single parameter
            calculation.

        Raises
        ------
        ValueError
            If a common multivariate distribution is given in
            Parameters.distribution and not all uncertain parameters are used.
        ValueError
            If `method` not one of "pc", "mc" or "custom".
        ValueError
            If `pc_method` not one of "collocation", "spectral" or "custom".
        NotImplementedError
            If custom method or custom pc method is chosen and have not been
            implemented.

        Notes
        -----
        Which method to choose is problem dependent, but as long as the number of
        uncertain parameters is low (less than around 20 uncertain parameters)
        polynomial chaos methods are much faster than Monte Carlo methods.
        Above this Monte Carlo methods are the best.

        For polynomial chaos, the pseudo-spectral method is faster than point
        collocation, but has lower stability. We therefore generally recommend
        the point collocation method.

        The model and feature do not necessarily give results for each
        node. The collocation method and quasi-Monte Carlo methods are robust
        towards missing values as long as the number of results that remain is
        high enough. The pseudo-spectral method on the other hand, is sensitive
        to missing values, so `allow_incomplete` should be used with care in
        that case.

        In the quasi-Monte Carlo method we quasi-randomly draw
        ``(nr_mc_samples/2)*(nr_uncertain_parameters + 2)``
        (nr_mc_samples=10**4 by default) parameter samples using Saltelli's
        sampling scheme. We require this number of samples to be able to calculate
        the Sobol indices. We evaluate the model for each of these parameter
        samples and calculate the features from each of the model results. This
        step is performed in parallel to speed up the calculations. Then we use
        `nr_mc_samples` of the model and feature results to calculate the
        mean, variance, and 5th and 95th percentile for the model and each
        feature. Lastly, we use all calculated model and each feature results to
        calculate the Sobol indices using Saltellie's approach.

        The plots created are intended as quick way to get an overview of the
        results, and not to create publication ready plots. Custom plots of the
        data can easily be created by retrieving the data from the Data class.

        Changing the parameters of the polynomial chaos methods should be done
        with care, and implementing custom methods is only recommended for
        experts.

        See also
        --------
        uncertainpy.Parameters
        uncertainpy.Data
        uncertainpy.plotting.PlotUncertainty
        uncertainpy.core.UncertaintyCalculations.polynomial_chaos : Uncertainty quantification using polynomial chaos expansions
        uncertainpy.core.UncertaintyCalculations.monte_carlo : Uncertainty quantification using quasi-Monte Carlo methods
        uncertainpy.core.UncertaintyCalculations.create_PCE_custom : Requirements for create_PCE_custom
        uncertainpy.core.UncertaintyCalculations.custom_uncertainty_quantification : Requirements for custom_uncertainty_quantification
        """
        uncertain_parameters = self.uncertainty_calculations.convert_uncertain_parameters(uncertain_parameters)

        if method.lower() == "pc":
            if single:
                data = self.polynomial_chaos_single(uncertain_parameters=uncertain_parameters,
                                                    method=pc_method,
                                                    rosenblatt=rosenblatt,
                                                    polynomial_order=polynomial_order,
                                                    nr_collocation_nodes=nr_collocation_nodes,
                                                    quadrature_order=quadrature_order,
                                                    nr_pc_mc_samples=nr_pc_mc_samples,
                                                    allow_incomplete=allow_incomplete,
                                                    seed=seed,
                                                    plot=plot,
                                                    figure_folder=figure_folder,
                                                    figureformat=figureformat,
                                                    save=save,
                                                    save_samples=save_samples,
                                                    data_folder=data_folder,
                                                    filename=filename,
                                                    **custom_kwargs)

            else:
                data = self.polynomial_chaos(uncertain_parameters=uncertain_parameters,
                                             method=pc_method,
                                             rosenblatt=rosenblatt,
                                             polynomial_order=polynomial_order,
                                             nr_collocation_nodes=nr_collocation_nodes,
                                             quadrature_order=quadrature_order,
                                             nr_pc_mc_samples=nr_pc_mc_samples,
                                             allow_incomplete=allow_incomplete,
                                             seed=seed,
                                             plot=plot,
                                             figure_folder=figure_folder,
                                             figureformat=figureformat,
                                             save=save,
                                             save_samples=save_samples,
                                             data_folder=data_folder,
                                             filename=filename,
                                             **custom_kwargs)

        elif method.lower() == "mc":
            if single:
                data = self.monte_carlo_single(uncertain_parameters=uncertain_parameters,
                                               nr_samples=nr_mc_samples,
                                               plot=plot,
                                               figure_folder=figure_folder,
                                               figureformat=figureformat,
                                               save=save,
                                               data_folder=data_folder,
                                               save_samples=save_samples,
                                               filename=filename,
                                               seed=seed)


            else:
                data = self.monte_carlo(uncertain_parameters=uncertain_parameters,
                                        nr_samples=nr_mc_samples,
                                        plot=plot,
                                        figure_folder=figure_folder,
                                        figureformat=figureformat,
                                        save=save,
                                        data_folder=data_folder,
                                        filename=filename,
                                        save_samples=save_samples,
                                        seed=seed)


        elif method.lower() == "custom":
            data = self.custom_uncertainty_quantification(plot=plot,
                                                          figure_folder=figure_folder,
                                                          figureformat=figureformat,
                                                          save=save,
                                                          data_folder=data_folder,
                                                          filename=filename,
                                                          **custom_kwargs)

        else:
            raise ValueError("No method with name {}".format(method))

        return data


    def custom_uncertainty_quantification(self,
                                          plot="condensed_first",
                                          figure_folder="figures",
                                          figureformat=".png",
                                          save=True,
                                          data_folder="data",
                                          filename=None,
                                          **custom_kwargs):
        """
        Perform a custom  uncertainty quantification and sensitivity analysis,
        implemented by the user.

        Parameters
        ----------
        plot : {"condensed_first", "condensed_total", "condensed_no_sensitivity", "all", "evaluations", None}, optional
            Type of plots to be created.
            "condensed_first" is a subset of the most important plots and
            only plots each result once, and contains plots of the first order
            Sobol indices. "condensed_total" is similar, but with the
            total order Sobol indices, and "condensed_no_sensitivity" is the
            same without any Sobol indices plotted. "all" creates every plot.
            "evaluations" plots the model and feature evaluations. None plots
            nothing.
            Default is "condensed_first".
        figure_folder : str, optional
            Name of the folder where to save all figures.
            Default is "figures".
        figureformat : str
            The figure format to save the plots in. Supports all formats in
            matplolib.
            Default is ".png".
        save : bool, optional
            If the data should be saved. Default is True.
        data_folder : str, optional
            Name of the folder where to save the data.
            Default is "data".
        filename : {None, str}, optional
            Name of the data file. If None the model name is used.
            Default is None.
        **custom_kwargs
            Any number of arguments for the custom uncertainty quantification.

        Raises
        ------
        NotImplementedError
            If the custom uncertainty quantification method have not been
            implemented.

        Notes
        -----
        For details on how to implement the custom uncertainty quantification
        method see UncertaintyCalculations.custom_uncertainty_quantification.

        The plots created are intended as quick way to get an overview of the
        results, and not to create publication ready plots. Custom plots of the
        data can easily be created by retrieving the data from the Data class.

        See also
        --------
        uncertainpy.plotting.PlotUncertainty
        uncertainpy.Parameters
        uncertainpy.core.UncertaintyCalculations.custom_uncertainty_quantification : Requirements for custom_uncertainty_quantification
        """

        self.data = self.uncertainty_calculations.custom_uncertainty_quantification(**custom_kwargs)

        self.data.backend = self.backend


        if filename is None:
            filename = self.model.name

        if save:
            self.save(filename, folder=data_folder)

        self.plot(type=plot,
                  folder=figure_folder,
                  figureformat=figureformat)

        return self.data


    def polynomial_chaos(self,
                         method="collocation",
                         rosenblatt="auto",
                         uncertain_parameters=None,
                         polynomial_order=4,
                         nr_collocation_nodes=None,
                         quadrature_order=None,
                         nr_pc_mc_samples=10**4,
                         allow_incomplete=True,
                         seed=None,
                         plot="condensed_first",
                         figure_folder="figures",
                         figureformat=".png",
                         save=True,
                         save_samples=False,
                         data_folder="data",
                         filename=None,
                         **custom_kwargs):
        """
        Perform an uncertainty quantification and sensitivity analysis
        using polynomial chaos expansions.

        Parameters
        ----------
        method : {"collocation", "spectral", "custom"}, optional
            The method to use when creating the polynomial chaos approximation,
            if the polynomial chaos method is chosen. "collocation" is the
            point collocation method "spectral" is pseudo-spectral projection,
            and "custom" is the custom polynomial method.
            Default is "collocation".
        rosenblatt : {"auto", bool}, optional
            If the Rosenblatt transformation should be used. The Rosenblatt
            transformation must be used if the uncertain parameters have
            dependent variables. If "auto" the Rosenblatt transformation is used
            if there are dependent parameters, and it is not used of the
            parameters have independent distributions. Default is "auto".
        uncertain_parameters : {None, str, list}, optional
            The uncertain parameter(s) to use when performing the uncertainty
            quantification. If None, all uncertain parameters are used.
            Default is None.
        polynomial_order : int, optional
            The polynomial order of the polynomial approximation.
            Default is 4.
        nr_collocation_nodes : {int, None}, optional
            The number of collocation nodes to choose, if polynomial chaos with
            point collocation is used. If None,
            `nr_collocation_nodes` = 2* number of expansion factors + 2.
            Default is None.
        quadrature_order : {int, None}, optional
            The order of the Leja quadrature method, if polynomial chaos with
            pseudo-spectral projection is used. If None,
            ``quadrature_order = polynomial_order + 2``.
            Default is None.
        nr_pc_mc_samples : int, optional
            Number of samples for the Monte Carlo sampling of the polynomial
            chaos approximation, if the polynomial chaos method is chosen.
        allow_incomplete : bool, optional
            If the polynomial approximation should be performed for features or
            models with incomplete evaluations.
            Default is True.
        seed : int, optional
            Set a random seed. If None, no seed is set.
            Default is None.
        plot : {"condensed_first", "condensed_total", "condensed_no_sensitivity", "all", "evaluations", None}, optional
            Type of plots to be created.
            "condensed_first" is a subset of the most important plots and
            only plots each result once, and contains plots of the first order
            Sobol indices. "condensed_total" is similar, but with the
            total order Sobol indices, and "condensed_no_sensitivity" is the
            same without any Sobol indices plotted. "all" creates every plot.
            "evaluations" plots the model and feature evaluations. None plots
            nothing.
            Default is "condensed_first".
        figure_folder : str, optional
            Name of the folder where to save all figures.
            Default is "figures".
        figureformat : str
            The figure format to save the plots in. Supports all formats in
            matplolib.
            Default is ".png".
        save : bool, optional
            If the data should be saved. Default is True.
        data_folder : str, optional
            Name of the folder where to save the data.
            Default is "data".
        filename : {None, str}, optional
            Name of the data file. If None the model name is used.
            Default is None.
        **custom_kwargs
            Any number of arguments for the custom polynomial chaos method,
            ``create_PCE_custom``.

        Returns
        -------
        data : Data
            A data object that contains the results from the uncertainty quantification.
            Contains all model and feature evaluations, as well as all calculated
            statistical metrics.

        Raises
        ------
        ValueError
            If a common multivariate distribution is given in
            Parameters.distribution and not all uncertain parameters are used.
        ValueError
            If `method` not one of "collocation", "spectral" or "custom".
        NotImplementedError
            If custom pc method is chosen and have not been implemented.

        Notes
        -----
        Which method to choose is problem dependent, but as long as the number of
        uncertain parameters is low (less than around 20 uncertain parameters)
        polynomial chaos methods are much faster than Monte Carlo methods.
        Above this Monte Carlo methods are the best.

        For polynomial chaos, the pseudo-spectral method is faster than point
        collocation, but has lower stability. We therefore generally recommend
        the point collocation method.

        The model and feature do not necessarily give results for each
        node. The collocation method are robust towards missing values as long
        as the number of results that remain is high enough. The pseudo-spectral
        method on the other hand, is sensitive to missing values, so
        `allow_incomplete` should be used with care in that case.

        The plots created are intended as quick way to get an overview of the
        results, and not to create publication ready plots. Custom plots of the
        data can easily be created by retrieving the data from the Data class.

        Changing the parameters of the polynomial chaos methods should be done
        with care, and implementing custom methods is only recommended for
        experts.

        See also
        --------
        uncertainpy.Data
        uncertainpy.Parameters
        uncertainpy.plotting.PlotUncertainty
        uncertainpy.core.UncertaintyCalculations.polynomial_chaos : Uncertainty quantification using polynomial chaos expansions
        uncertainpy.core.UncertaintyCalculations.create_PCE_custom : Requirements for create_PCE_custom
        """
        uncertain_parameters = self.uncertainty_calculations.convert_uncertain_parameters(uncertain_parameters)

        if len(uncertain_parameters) > 20:
            raise RuntimeWarning("The number of uncertain parameters is high."
                                 + "The Monte-Carlo method might be faster.")


        self.data = self.uncertainty_calculations.polynomial_chaos(
            method=method,
            rosenblatt=rosenblatt,
            uncertain_parameters=uncertain_parameters,
            polynomial_order=polynomial_order,
            nr_collocation_nodes=nr_collocation_nodes,
            quadrature_order=quadrature_order,
            nr_pc_mc_samples=nr_pc_mc_samples,
            allow_incomplete=allow_incomplete,
            save_samples=save_samples,
            seed=seed,
            **custom_kwargs
            )

        self.data.backend = self.backend

        if filename is None:
            filename = self.model.name

        if save:
            self.save(filename, folder=data_folder)

        self.plot(type=plot,
                  folder=figure_folder,
                  figureformat=figureformat)

        return self.data


    def monte_carlo(self,
                    uncertain_parameters=None,
                    nr_samples=10**4,
                    seed=None,
                    plot="condensed_first",
                    figure_folder="figures",
                    figureformat=".png",
                    save=True,
                    data_folder="data",
                    save_samples=False,
                    filename=None):
        """
        Perform an uncertainty quantification using the quasi-Monte Carlo method.

        Parameters
        ----------
        uncertain_parameters : {None, str, list}, optional
            The uncertain parameter(s) to use when performing the uncertainty
            quantification. If None, all uncertain parameters are used.
            Default is None.
        nr_samples : int, optional
            Number of samples for the quasi-Monte Carlo sampling.
            `nr_samples` is used for the uncertainty
            quantification and ``(nr_samples/2)*(nr_uncertain_parameters + 2)``
            samples is used for the sensitivity analysis. Default `nr_samples`
            is 10**4.
        seed : int, optional
            Set a random seed. If None, no seed is set.
            Default is None.
        plot : {"condensed_first", "condensed_total", "condensed_no_sensitivity", "all", "evaluations", None}, optional
            Type of plots to be created.
            "condensed_first" is a subset of the most important plots and
            only plots each result once, and contains plots of the first order
            Sobol indices. "condensed_total" is similar, but with the
            total order Sobol indices, and "condensed_no_sensitivity" is the
            same without any Sobol indices plotted. "all" creates every plot.
            "evaluations" plots the model and feature evaluations. None plots
            nothing.
            Default is "condensed_first".
        figure_folder : str, optional
            Name of the folder where to save all figures.
            Default is "figures".
        figureformat : str
            The figure format to save the plots in. Supports all formats in
            matplolib.
            Default is ".png".
        save : bool, optional
            If the data should be saved. Default is True.
        data_folder : str, optional
            Name of the folder where to save the data.
            Default is "data".
        filename : {None, str}, optional
            Name of the data file. If None the model name is used.
            Default is None.

        Returns
        -------
        data : Data
            A data object that contains the results from the uncertainty quantification.
            Contains all model and feature evaluations, as well as all calculated
            statistical metrics.

        Raises
        ------
        ValueError
            If a common multivariate distribution is given in
            Parameters.distribution and not all uncertain parameters are used.

        Notes
        -----
        Which method to choose is problem dependent, but as long as the number of
        uncertain parameters is low (less than around 20 uncertain parameters)
        polynomial chaos methods are much faster than Monte Carlo methods.
        Above this Monte Carlo methods are the best.

        In the quasi-Monte Carlo method we quasi-randomly draw
        ``(nr_samples/2)*(nr_uncertain_parameters + 2)``
        (nr_samples=10**4 by default) parameter samples using Saltelli's
        sampling scheme. We require this number of samples to be able to calculate
        the Sobol indices. We evaluate the model for each of these parameter
        samples and calculate the features from each of the model results. This
        step is performed in parallel to speed up the calculations. Then we use
        `nr_samples` of the model and feature results to calculate the
        mean, variance, and 5th and 95th percentile for the model and each
        feature. Lastly, we use all calculated model and each feature results to
        calculate the Sobol indices using Saltellie's approach.

        The plots created are intended as quick way to get an overview of the
        results, and not to create publication ready plots. Custom plots of the
        data can easily be created by retrieving the data from the Data class.

        Sensitivity analysis is currently not yet available for the quasi-Monte
        Carlo method.

        See also
        --------
        uncertainpy.Data
        uncertainpy.Parameters
        uncertainpy.plotting.PlotUncertainty
        uncertainpy.core.UncertaintyCalculations.monte_carlo : Uncertainty quantification using quasi-Monte Carlo methods
        """
        uncertain_parameters = self.uncertainty_calculations.convert_uncertain_parameters(uncertain_parameters)


        self.data = self.uncertainty_calculations.monte_carlo(uncertain_parameters=uncertain_parameters,
                                                              nr_samples=nr_samples,
                                                              save_samples=save_samples,
                                                              seed=seed)

        self.data.backend = self.backend

        if filename is None:
           filename = self.model.name

        if save:
            self.save(filename, folder=data_folder)

        self.plot(type=plot,
                  folder=figure_folder,
                  figureformat=figureformat)

        return self.data


    def polynomial_chaos_single(self,
                                method="collocation",
                                rosenblatt="auto",
                                polynomial_order=4,
                                uncertain_parameters=None,
                                nr_collocation_nodes=None,
                                quadrature_order=None,
                                nr_pc_mc_samples=10**4,
                                allow_incomplete=True,
                                seed=None,
                                plot="condensed_first",
                                figure_folder="figures",
                                figureformat=".png",
                                save=True,
                                save_samples=False,
                                data_folder="data",
                                filename=None):
        """
        Perform an uncertainty quantification and sensitivity analysis for a
        single parameter at the time using polynomial chaos expansions.

        Parameters
        ----------
        method : {"collocation", "spectral", "custom"}, optional
            The method to use when creating the polynomial chaos approximation,
            if the polynomial chaos method is chosen. "collocation" is the
            point collocation method "spectral" is pseudo-spectral projection,
            and "custom" is the custom polynomial method.
            Default is "collocation".
        rosenblatt : {"auto", bool}, optional
            If the Rosenblatt transformation should be used. The Rosenblatt
            transformation must be used if the uncertain parameters have
            dependent variables. If "auto" the Rosenblatt transformation is used
            if there are dependent parameters, and it is not used of the
            parameters have independent distributions. Default is "auto".
        uncertain_parameters : {None, str, list}, optional
            The uncertain parameter(s) to performing the uncertainty
            quantification for. If None, all uncertain parameters are used.
            Default is None.
        polynomial_order : int, optional
            The polynomial order of the polynomial approximation.
            Default is 4.
        nr_collocation_nodes : {int, None}, optional
            The number of collocation nodes to choose, if polynomial chaos with
            point collocation is used. If None,
            `nr_collocation_nodes` = 2* number of expansion factors + 2.
            Default is None.
        quadrature_order : {int, None}, optional
            The order of the Leja quadrature method, if polynomial chaos with
            pseudo-spectral projection is used. If None,
            ``quadrature_order = polynomial_order + 2``.
            Default is None.
        nr_pc_mc_samples : int, optional
            Number of samples for the Monte Carlo sampling of the polynomial
            chaos approximation, if the polynomial chaos method is chosen.
        allow_incomplete : bool, optional
            If the polynomial approximation should be performed for features or
            models with incomplete evaluations.
            Default is True.
        seed : int, optional
            Set a random seed. If None, no seed is set.
            Default is None.
        plot : {"condensed_first", "condensed_total", "condensed_no_sensitivity", "all", "evaluations", None}, optional
            Type of plots to be created.
            "condensed_first" is a subset of the most important plots and
            only plots each result once, and contains plots of the first order
            Sobol indices. "condensed_total" is similar, but with the
            total order Sobol indices, and "condensed_no_sensitivity" is the
            same without any Sobol indices plotted. "all" creates every plot.
            "evaluations" plots the model and feature evaluations. None plots
            nothing.
            Default is "condensed_first".
        figure_folder : str, optional
            Name of the folder where to save all figures.
            Default is "figures".
        figureformat : str
            The figure format to save the plots in. Supports all formats in
            matplolib.
            Default is ".png".
        save : bool, optional
            If the data should be saved. Default is True.
        data_folder : str, optional
            Name of the folder where to save the data.
            Default is "data".
        filename : {None, str}, optional
            Name of the data file. If None the model name is used.
            Default is None.
        **custom_kwargs
            Any number of arguments for the custom polynomial chaos method,
            ``create_PCE_custom``.

        Returns
        -------
        data_dict : dict
            A dictionary that contains the data for each single parameter
            calculation.

        Raises
        ------
        ValueError
            If a common multivariate distribution is given in
            Parameters.distribution and not all uncertain parameters are used.
        ValueError
            If `method` not one of "collocation", "spectral" or "custom".
        NotImplementedError
            If custom pc method is chosen and have not been implemented.

        Notes
        -----
        Which method to choose is problem dependent, but as long as the number of
        uncertain parameters is low (less than around 20 uncertain parameters)
        polynomial chaos methods are much faster than Monte Carlo methods.
        Above this Monte Carlo methods are the best.

        For polynomial chaos, the pseudo-spectral method is faster than point
        collocation, but has lower stability. We therefore generally recommend
        the point collocation method.

        The model and feature do not necessarily give results for each
        node. The collocation method are robust towards missing values as long
        as the number of results that remain is high enough. The pseudo-spectral
        method on the other hand, is sensitive to missing values, so
        `allow_incomplete` should be used with care in that case.

        The plots created are intended as quick way to get an overview of the
        results, and not to create publication ready plots. Custom plots of the
        data can easily be created by retrieving the data from the Data class.

        Changing the parameters of the polynomial chaos methods should be done
        with care, and implementing custom methods is only recommended for
        experts.

        See also
        --------
        uncertainpy.Data
        uncertainpy.Parameters
        uncertainpy.plotting.PlotUncertainty
        uncertainpy.core.UncertaintyCalculations.polynomial_chaos : Uncertainty quantification using polynomial chaos expansions
        uncertainpy.core.UncertaintyCalculations.create_PCE_custom : Requirements for create_PCE_custom

        """
        logger = get_logger(self)

        uncertain_parameters = self.uncertainty_calculations.convert_uncertain_parameters(uncertain_parameters)

        for parameter in self.parameters:
            if parameter.value is None:
                raise ValueError("Parameter.value must be set for each parameter when using single=True.")

        if filename is None:
            filename = self.model.name

        if seed is not None:
            np.random.seed(seed)

        data_dict = {}

        for uncertain_parameter in uncertain_parameters:
            logger.info("Running for " + uncertain_parameter)

            data = self.uncertainty_calculations.polynomial_chaos(
                uncertain_parameters=uncertain_parameter,
                method=method,
                rosenblatt=rosenblatt,
                polynomial_order=polynomial_order,
                nr_collocation_nodes=nr_collocation_nodes,
                quadrature_order=quadrature_order,
                nr_pc_mc_samples=nr_pc_mc_samples,
                save_samples=save_samples,
                allow_incomplete=allow_incomplete)

            data.backend = self.backend
            data.seed = seed
            self.data = data

            data_dict[uncertain_parameter] = data

        self.data = data_dict

        if save:
            self.save(filename, folder=data_folder)

        self.plot(type=plot,
                  folder=figure_folder,
                  figureformat=figureformat)

        return data_dict


    def monte_carlo_single(self,
                           uncertain_parameters=None,
                           nr_samples=10**4,
                           seed=None,
                           plot="condensed_first",
                           save=True,
                           data_folder="data",
                           figure_folder="figures",
                           figureformat=".png",
                           save_samples=False,
                           filename=None):
        """
        Perform an uncertainty quantification for a single parameter at the time
        using the quasi-Monte Carlo method.

        Parameters
        ----------
        uncertain_parameters : {None, str, list}, optional
            The uncertain parameter(s) to use when performing the uncertainty
            quantification. If None, all uncertain parameters are used.
            Default is None.
        nr_samples : int, optional
            Number of samples for the quasi-Monte Carlo sampling.
            `nr_samples` is used for the uncertainty
            quantification and ``(nr_samples/2)*(nr_uncertain_parameters + 2)``
            samples is used for the sensitivity analysis. Default `nr_samples`
            is 10**4.
        seed : int, optional
            Set a random seed. If None, no seed is set.
            Default is None.
        plot : {"condensed_first", "condensed_total", "condensed_no_sensitivity", "all", "evaluations", None}, optional
            Type of plots to be created.
            "condensed_first" is a subset of the most important plots and
            only plots each result once, and contains plots of the first order
            Sobol indices. "condensed_total" is similar, but with the
            total order Sobol indices, and "condensed_no_sensitivity" is the
            same without any Sobol indices plotted. "all" creates every plot.
            "evaluations" plots the model and feature evaluations. None plots
            nothing.
            Default is "condensed_first".
        figure_folder : str, optional
            Name of the folder where to save all figures.
            Default is "figures".
        figureformat : str
            The figure format to save the plots in. Supports all formats in
            matplolib.
            Default is ".png".
        save : bool, optional
            If the data should be saved. Default is True.
        data_folder : str, optional
            Name of the folder where to save the data.
            Default is "data".
        filename : {None, str}, optional
            Name of the data file. If None the model name is used.
            Default is None.

        Returns
        -------
        data_dict : dict
            A dictionary that contains the data objects for each single parameter
            calculation.

        Raises
        ------
        ValueError
            If a common multivariate distribution is given in
            Parameters.distribution and not all uncertain parameters are used.

        Notes
        -----
        Which method to choose is problem dependent, but as long as the number of
        uncertain parameters is low (less than around 20 uncertain parameters)
        polynomial chaos methods are much faster than Monte Carlo methods.
        Above this Monte Carlo methods are the best.

        In the quasi-Monte Carlo method we quasi-randomly draw
        ``(nr_samples/2)*(nr_uncertain_parameters + 2)``
        (nr_samples=10**4 by default) parameter samples using Saltelli's
        sampling scheme. We require this number of samples to be able to calculate
        the Sobol indices. We evaluate the model for each of these parameter
        samples and calculate the features from each of the model results. This
        step is performed in parallel to speed up the calculations. Then we use
        `nr_samples` of the model and feature results to calculate the
        mean, variance, and 5th and 95th percentile for the model and each
        feature. Lastly, we use all calculated model and each feature results to
        calculate the Sobol indices using Saltellie's approach.

        The plots created are intended as quick way to get an overview of the
        results, and not to create publication ready plots. Custom plots of the
        data can easily be created by retrieving the data from the Data class.

        Sensitivity analysis is currently not yet available for the quasi-Monte
        Carlo method.

        See also
        --------
        uncertainpy.Data
        uncertainpy.plotting.PlotUncertainty
        uncertainpy.Parameters
        uncertainpy.core.UncertaintyCalculations.monte_carlo : Uncertainty quantification using quasi-Monte Carlo methods
        """
        logger = get_logger(self)

        uncertain_parameters = self.uncertainty_calculations.convert_uncertain_parameters(uncertain_parameters)

        if filename is None:
            filename = self.model.name

        if seed is not None:
            np.random.seed(seed)

        data_dict = {}
        for uncertain_parameter in uncertain_parameters:
            logger.info("Running MC for " + uncertain_parameter)

            data = self.uncertainty_calculations.monte_carlo(uncertain_parameters=uncertain_parameter,
                                                             nr_samples=nr_samples, save_samples=save_samples)

            data.backend = self.backend
            data.seed = seed

            data_dict[uncertain_parameter] = data

        self.data = data_dict

        if save:
            self.save(filename, folder=data_folder)

        self.plot(type=plot,
                  folder=figure_folder,
                  figureformat=figureformat)

        return data_dict


    def save(self, filename, folder="data"):
        """
        Save ``data`` to disk.

        Parameters
        ----------
        filename : str
            Name of the data file.
        folder : str, optional
            The folder to store the data in. Creates the folder if it does not
            exist. Default is "/data".

        See also
        --------
        uncertainpy.Data : Data class
        """
        if not os.path.isdir(folder):
            os.makedirs(folder)

        logger = get_logger(self)

        fileextension = ""
        if self.backend == "auto":
            if filename.endswith(".h5"):
                fileextension = ".h5"
                filename = filename.strip(".h5")
            elif filename.endswith(".exdir"):
                fileextension = ".exdir"
                filename = filename.strip(".exdir")
            else:
                fileextension = ".h5"

        elif self.backend == "hdf5":
            fileextension =  ".h5"
            filename = filename.strip(".h5")
        elif self.backend == "exdir":
            fileextension = ".exdir"
            filename = filename.strip(".exdir")

        # To save dict of single parameter runs
        if isinstance(self.data, dict):
            for uncertain_parameter in self.data:
                tmp_filename = "{}_{}".format(
                    filename,
                    uncertain_parameter
                )

                save_path = os.path.join(folder, tmp_filename + fileextension)

                logger.info("Saving data as: {}".format(save_path))

                self.data[uncertain_parameter].save(save_path)

        else:
            save_path = os.path.join(folder, filename + fileextension)

            logger.info("Saving data as: {}".format(save_path))

            self.data.save(save_path)




    def load(self, filename):
        """
        Load data from disk.

        Parameters
        ----------
        filename : str
            Name of the stored data file.

        See also
        --------
        uncertainpy.Data : Data class
        """
        self.data = Data(filename)


    def plot(self,
             type="condensed_first",
             folder="figures",
             figureformat=".png"):
        """
        Create plots for the results of the uncertainty quantification and
        sensitivity analysis. ``self.data`` must exist and contain the results.

        Parameters
        ----------
        data : Data
            A data object that contains the results from the uncertainty quantification.
        type : {"condensed_first", "condensed_total", "condensed_no_sensitivity", "all", "evaluations", None}, optional
            Type of plots to be created.
            "condensed_first" is a subset of the most important plots and
            only plots each result once, and contains plots of the first order
            Sobol indices. "condensed_total" is similar, but with the
            total order Sobol indices, and "condensed_no_sensitivity" is the
            same without any Sobol indices plotted. "all" creates every plot.
            "evaluations" plots the model and feature evaluations. None plots
            nothing. Default is "condensed_first".
        folder : str
            Name of the folder where to save all figures. Default is "figures".
        figureformat : str
            The figure format to save the plots in. Supports all formats in
            matplolib. Default is ".png".

        Notes
        -----
        These plots are intended as quick way to get an overview of the results,
        and not to create publication ready plots. Custom plots of the data can
        easily be created by retrieving the data from the Data class.

        See also
        --------
        uncertainpy.Data
        uncertainpy.plotting.PlotUncertainty
        """

        def plot(type):
            if type.lower() == "condensed_first":
                self.plotting.plot_condensed(sensitivity="sobol_first")

            elif type.lower() == "condensed_total":
                self.plotting.plot_condensed(sensitivity="sobol_total")

            elif type.lower() == "condensed_no_sensitivity":
                self.plotting.plot_condensed(sensitivity=None)

            elif type.lower() == "all":
                self.plotting.plot_all_sensitivities()
                self.plotting.all_evaluations()

            elif type.lower() == "evaluations":
                self.plotting.all_evaluations()

            else:
                raise ValueError('type must one of: "condensed_first", '
                                 '"condensed_total", "condensed_no_sensitivity" '
                                 '"all", "evaluations", None, not {}'.format(type))


        if type is None:
            return
        else:
            self.plotting.figureformat = figureformat

            # To plot dict of single parameter runs
            if isinstance(self.data, dict):
                for uncertain_parameter in self.data:
                    tmp_folder = os.path.join(folder, uncertain_parameter)

                    self.plotting.folder = tmp_folder
                    self.plotting.set_data(self.data[uncertain_parameter])

                    plot(type)

            else:
                self.plotting.folder = folder
                self.plotting.set_data(self.data)

                plot(type)

