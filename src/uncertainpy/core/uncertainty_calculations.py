from __future__ import absolute_import, division, print_function, unicode_literals

import six
import numpy as np
from tqdm import tqdm
import chaospy as cp
import types
from SALib.sample import saltelli
from SALib.analyze.sobol import first_order, total_order

from .run_model import RunModel
from .base import ParameterBase
from ..utils.utility import contains_nan
from ..utils.logger import get_logger




class UncertaintyCalculations(ParameterBase):
    """
    Perform the calculations for the uncertainty quantification and
    sensitivity analysis.

    This class performs the calculations for the uncertainty quantification and
    sensitivity analysis of the model and features. It implements both
    quasi-Monte Carlo methods and polynomial chaos expansions using either
    point collocation or pseudo-spectral method. Both of the polynomial chaos
    expansion methods have support for the rosenblatt transformation to handle
    dependent variables.

    Parameters
    ----------
    model : {None, Model or Model subclass instance, model function}, optional
        Model to perform uncertainty quantification on. For requirements see
        Model.run.
        Default is None.
    parameters: {dict {name: parameter_object}, dict of {name: value or Chaospy distribution}, ...], list of Parameter instances, list [[name, value or Chaospy distribution], ...], list [[name, value, Chaospy distribution or callable that returns a Chaospy distribution],...],}
        List or dictionary of the parameters that should be created.
        On the form ``parameters =``

            * ``{name_1: parameter_object_1, name: parameter_object_2, ...}``
            * ``{name_1:  value_1 or Chaospy distribution, name_2:  value_2 or Chaospy distribution, ...}``
            * ``[parameter_object_1, parameter_object_2, ...]``,
            * ``[[name_1, value_1 or Chaospy distribution], ...]``.
            * ``[[name_1, value_1, Chaospy distribution or callable that returns a Chaospy distribution], ...]``

    features : {None, Features or Features subclass instance, list of feature functions}, optional
        Features to calculate from the model result.
        If None, no features are calculated.
        If list of feature functions, all will be calculated.
        Default is None.
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
        than this level is ignored. If None, no logging to file is performed.
        Default logger level is "info".

    Attributes
    ----------
    model : Model or Model subclass
        The model to perform uncertainty quantification on.
    parameters : Parameters
        The uncertain parameters.
    features : Features or Features subclass
        The features of the model to perform uncertainty quantification on.
    runmodel : RunModel
        Runmodel object responsible for evaluating the model and calculating features.

    See Also
    --------
    uncertainpy.features.Features
    uncertainpy.Parameter
    uncertainpy.Parameters
    uncertainpy.models.Model
    uncertainpy.core.RunModel
    uncertainpy.models.Model.run : Requirements for the model run function.
    """
    def __init__(self,
                 model=None,
                 parameters=None,
                 features=None,
                 create_PCE_custom=None,
                 custom_uncertainty_quantification=None,
                 CPUs="max",
                 logger_level="info"):


        self.runmodel = RunModel(model=model,
                                 parameters=parameters,
                                 features=features,
                                 logger_level=logger_level,
                                 CPUs=CPUs)


        if create_PCE_custom is not None:
            self.create_PCE_custom = create_PCE_custom

        if custom_uncertainty_quantification is not None:
            self.custom_uncertainty_quantification = custom_uncertainty_quantification

        super(UncertaintyCalculations, self).__init__(parameters=parameters,
                                                      model=model,
                                                      features=features,
                                                      logger_level=logger_level)


    @ParameterBase.features.setter
    def features(self, new_features):
        ParameterBase.features.fset(self, new_features)

        self.runmodel.features = self.features


    @ParameterBase.model.setter
    def model(self, new_model):
        ParameterBase.model.fset(self, new_model)

        self.runmodel.model = self.model


    @ParameterBase.parameters.setter
    def parameters(self, new_parameters):
        ParameterBase.parameters.fset(self, new_parameters)

        self.runmodel.parameters = self.parameters




    def convert_uncertain_parameters(self, uncertain_parameters=None):
        """
        Converts uncertain_parameter(s) to a list of uncertain parameter(s), and
        checks if it is a legal set of uncertain parameter(s).

        Parameters
        ----------
        uncertain_parameters : {None, str, list}, optional
            The name(s) of the uncertain parameters to use. If None, a list of
            all uncertain parameters are returned.
            Default is None.

        Returns
        -------
        uncertain_parameters : list
            A list with the name of all uncertain parameters.

        Raises
        ------
        ValueError
            If a common multivariate distribution is given in
            Parameters.distribution and not all uncertain parameters are used.

        See Also
        --------
        uncertainpy.Parameters
        """
        if isinstance(uncertain_parameters, six.string_types):
            uncertain_parameters = [uncertain_parameters]

        if self.parameters.distribution is None:
            if uncertain_parameters is None:
                uncertain_parameters = self.parameters.get_from_uncertain("name")

        else:
            if uncertain_parameters is None:
                uncertain_parameters = self.parameters.get("name")
            elif sorted(uncertain_parameters) != sorted(self.parameters.get("name")):
                 raise ValueError("A common multivariate distribution is given, " +
                                  "and all uncertain parameters must be used. " +
                                  "Set uncertain_parameters to None or a list of all " +
                                  "uncertain parameters.")

        return uncertain_parameters


    def create_distribution(self, uncertain_parameters=None):
        """
        Create a joint multivariate distribution for the selected parameters from
        univariate distributions.

        Parameters
        ----------
        uncertain_parameters : {None, str, list}, optional
            The uncertain parameter(s) to use when creating the joint multivariate
            distribution. If None, the joint multivariate distribution for all
            uncertain parameters is created.
            Default is None.

        Returns
        -------
        distribution : chaospy.Dist
            The joint multivariate distribution for the given parameters.

        Raises
        ------
        ValueError
            If a common multivariate distribution is given in
            Parameters.distribution and not all uncertain parameters are used.

        Notes
        -----
        If a multivariate distribution is defined in the Parameters.distribution,
        that multivariate distribution is returned. Otherwise the joint
        multivariate distribution for the selected parameters is created from
        the univariate distributions.

        See also
        --------
        uncertainpy.Parameters
        """
        uncertain_parameters = self.convert_uncertain_parameters(uncertain_parameters)

        if self.parameters.distribution is None:
            parameter_distributions = self.parameters.get("distribution", uncertain_parameters)

            distribution = cp.J(*parameter_distributions)
        else:
            distribution = self.parameters.distribution

        return distribution


    def dependent(self, distribution):
        """
        Check if a distribution is dependent or not.

        Parameters
        ----------
        distribution : chaospy.Dist
            A Chaospy probability distribution.

        Returns
        -------
        dependent : bool
            True if the distribution is dependent, False if is independent.
        """
        # New property added in Chaospy, so the dependent method is
        # kept for legacy
        return distribution.stochastic_dependent


    def create_mask(self, evaluations):
        """
        Mask evaluations that do not give results (anything but np.nan or None).

        Parameters
        ----------
        evaluations : array_like
            Evaluations for the model.

        Returns
        -------
        masked_evaluations : list
            The evaluations that have results (not numpy.nan or None).
        mask : boolean array
            The mask itself, used to create the masked arrays.
        """
        masked_evaluations = []
        mask = np.ones(len(evaluations), dtype=bool)

        for i, result in enumerate(evaluations):
            # if np.any(np.isnan(result)):
            if contains_nan(result):
                mask[i] = False
            else:
                masked_evaluations.append(result)

        return masked_evaluations, mask


    def create_masked_evaluations(self, data, feature):
        """
        Mask all model and feature evaluations that do not give results
        (anything but np.nan) and the corresponding nodes.

        Parameters
        ----------
        data : Data
            A Data object with evaluations for the model and each feature.
            Must contain `data[feature].evaluations`.
        feature : str
            Name of the feature or model to mask.

        Returns
        -------
        masked_evaluations : list
            The evaluations that have results (not numpy.nan or None).
        mask : boolean array
            The mask itself, used to create the masked arrays.
        """
        if feature not in data:
            raise AttributeError("Error: {} is not a feature".format(feature))

        masked_evaluations, mask = self.create_mask(data[feature].evaluations)

        if not np.all(mask):
            logger = get_logger(self)
            logger.warning("{}: only yields ".format(feature) +
                           "results for {}/{} ".format(sum(mask), len(mask)) +
                           "parameter combinations.")


        return masked_evaluations, mask



    def create_masked_nodes(self, data, feature, nodes):
        """
        Mask all model and feature evaluations that do not give results
        (anything but np.nan) and the corresponding nodes.

        Parameters
        ----------
        data : Data
            A Data object with evaluations for the model and each feature.
            Must contain `data[feature].evaluations`.
        feature : str
            Name of the feature or model to mask.
        nodes : array_like
            The nodes used to evaluate the model.

        Returns
        -------
        masked_evaluations : array_like
            The evaluations which have results.
        mask : boolean array
            The mask itself, used to create the masked arrays.
        masked_nodes : array_like
            The nodes that correspond to the evaluations with results.
        """
        masked_evaluations, mask = self.create_masked_evaluations(data, feature)

        if len(nodes.shape) > 1:
            masked_nodes = nodes[:, mask]
        else:
            masked_nodes = nodes[mask]

        return masked_evaluations, mask, masked_nodes



    def create_masked_nodes_weights(self, data, feature, nodes, weights):
        """
        Mask all model and feature evaluations that do not give results
        (anything but numpy.nan) and the corresponding nodes.

        Parameters
        ----------
        data : Data
            A Data object with evaluations for the model and each feature.
            Must contain `data[feature].evaluations`.
        nodes : array_like
            The nodes used to evaluate the model.
        feature : str
            Name of the feature or model to mask.
        weights : array_like
            Weights corresponding to each node.

        Returns
        -------
        masked_evaluations : array_like
            The evaluations which have results.
        mask : boolean array
            The mask itself, used to create the masked arrays.
        masked_nodes : array_like
            The nodes that correspond to the evaluations with results.
        masked_weights : array_like
            Masked weights that correspond to evaluations with results.
        """
        masked_evaluations, mask, masked_nodes = self.create_masked_nodes(data, feature, nodes)

        if len(weights.shape) > 1:
            masked_weights = weights[:, mask]
        else:
            masked_weights = weights[mask]

        return masked_evaluations, mask, masked_nodes, masked_weights


    def get_UQ_samples(self, method, uncertain_parameters, polynomial_order, quadrature_order=None, seed=None):
        if method not in ["spectral", "collocation", "MC"]:
            raise ValueError("The UQ method needs to be provided (spectral, collocation or MC)")
        uncertain_parameters = self.convert_uncertain_parameters(uncertain_parameters)
        distribution = self.create_distribution(uncertain_parameters=uncertain_parameters)
        if method == "spectral":
            if quadrature_order is None:
                raise ValueError("The quadrature_order must be an integer")
            P = cp.orth_ttr(polynomial_order, distribution)
            nodes, weights = cp.generate_quadrature(quadrature_order,
                                                    distribution,
                                                    rule="J",
                                                    sparse=True)
            return uncertain_parameters, distribution, P, nodes, weights
        elif method == "collocation":
            P = cp.orth_ttr(polynomial_order, distribution)
            nr_collocation_nodes = quadrature_order
            if nr_collocation_nodes is None:
                nr_collocation_nodes = 2 * len(P) + 2
            nodes = distribution.sample(nr_collocation_nodes, "M")
            return uncertain_parameters, distribution, P, nodes, None
        elif method == "MC":
            nr_samples = polynomial_order
            if nr_samples is None:
                raise ValueError("The nr_samples must be an integer")

            if seed is not None:
                np.random.seed(seed)

            problem = {
                "num_vars": len(uncertain_parameters),
                "names": uncertain_parameters,
                "bounds": [[0,1]]*len(uncertain_parameters)
            }

            # Create the Multivariate normal distribution
            dist_R = []
            for parameter in uncertain_parameters:
                dist_R.append(cp.Uniform())

            dist_R = cp.J(*dist_R)

            nr_sobol_samples = int(np.round(nr_samples/2.))

            nodes_R = saltelli.sample(problem, nr_sobol_samples, calc_second_order=False)

            nodes = distribution.inv(dist_R.fwd(nodes_R.transpose()))
            return uncertain_parameters, distribution, None, nodes, nr_sobol_samples


    def create_PCE_spectral(self,
                            uncertain_parameters=None,
                            polynomial_order=4,
                            quadrature_order=None,
                            allow_incomplete=True):
        """
        Create the polynomial approximation `U_hat` using pseudo-spectral
        projection.

        Parameters
        ----------
        uncertain_parameters : {None, str, list}, optional
            The uncertain parameter(s) to use when creating the polynomial
            approximation. If None, all uncertain parameters are used.
            Default is None.
        polynomial_order : int, optional
            The polynomial order of the polynomial approximation.
            Default is 4.
        quadrature_order : {int, None}, optional
            The order of the Leja quadrature method. If None,
            ``quadrature_order = polynomial_order + 2``.
            Default is None.
        allow_incomplete : bool, optional
            If the polynomial approximation should be performed for features or
            models with incomplete evaluations.
            Default is True.

        Returns
        -------
        U_hat : dict
            A dictionary containing the polynomial approximations for the
            model and each feature as chaospy.Poly objects.
        distribution : chaospy.Dist
            The multivariate distribution for the uncertain parameters.
        data : Data
            A data object containing the values from the model evaluation
            and feature calculations.

        Raises
        ------
        ValueError
            If a common multivariate distribution is given in
            Parameters.distribution and not all uncertain parameters are used.

        Notes
        -----
        The returned `data` should contain (but not necessarily) the following:

            1. ``data["model/features"].evaluations``
            2. ``data["model/features"].time``
            3. ``data["model/features"].labels``
            4. ``data.model_name``
            5. ``data.incomplete``
            6. ``data.method``
            7. ``data.errored``

        The model and feature do not necessarily give results for each
        node. The pseudo-spectral methods is sensitive to missing values, so
        `allow_incomplete` should be used with care.

        The polynomial chaos expansion method for uncertainty quantification
        approximates the model with a polynomial that follows specific
        requirements. This polynomial can be used to quickly calculate the
        uncertainty and sensitivity of the model.

        To create the polynomial chaos expansion we first find the polynomials
        using the three-therm recurrence relation if available,
        otherwise the discretized Stieltjes method is used. Then we use the
        pseudo-spectral projection to find the expansion coefficients for the
        model and each feature of the model.

        Pseudo-spectral projection is based on least squares minimization and
        finds the expansion coefficients through numerical integration. The
        integration uses a quadrature scheme with weights and nodes. We use Leja
        quadrature with Smolyak sparse grids to reduce the number of nodes
        required. For each of the nodes we evaluate the model and calculate the
        features, and the polynomial approximation is created from these results.

        See also
        --------
        uncertainpy.Data
        uncertainpy.Parameters
        """
        if quadrature_order is None:
            quadrature_order = polynomial_order + 2
        uncertain_parameters, distribution, P, nodes, weights = self.get_UQ_samples("spectral", uncertain_parameters, polynomial_order, quadrature_order=quadrature_order) 

        # Running the model
        data = self.runmodel.run(nodes, uncertain_parameters)

        # TODO: import data here instead of running

        data.method = "polynomial chaos expansion with the pseudo-spectral method. polynomial_order={}, quadrature_order={}".format(polynomial_order, quadrature_order)

        logger = get_logger(self)

        U_hat = {}
        # Calculate PC for each feature
        for feature in tqdm(data,
                            desc="Calculating PC for each feature",
                            total=len(data)):
            if feature == self.model.name and self.model.ignore:
                continue

            masked_evaluations, mask, masked_nodes, masked_weights = \
                self.create_masked_nodes_weights(data, feature, nodes, weights)


            if (np.all(mask) or allow_incomplete) and sum(mask) > 0:
                U_hat[feature] = cp.fit_quadrature(P, masked_nodes,
                                                   masked_weights, masked_evaluations)
            elif not allow_incomplete:
                logger.warning("{}: not all parameter combinations give results.".format(feature) +
                               " No uncertainty quantification is performed since allow_incomplete=False")

            else:
                logger.warning("{}: not all parameter combinations give results.".format(feature))

            if not np.all(mask):
                data.incomplete.append(feature)

        return U_hat, distribution, data


    def create_PCE_collocation(self,
                               uncertain_parameters=None,
                               polynomial_order=4,
                               nr_collocation_nodes=None,
                               allow_incomplete=True):
        """
        Create the polynomial approximation `U_hat` using pseudo-spectral
        projection.

        Parameters
        ----------
        uncertain_parameters : {None, str, list}, optional
            The uncertain parameter(s) to use when creating the polynomial
            approximation. If None, all uncertain parameters are used.
            Default is None.
        polynomial_order : int, optional
            The polynomial order of the polynomial approximation.
            Default is 4.
        nr_collocation_nodes : {int, None}, optional
            The number of collocation nodes to choose. If None,
            `nr_collocation_nodes` = 2* number of expansion factors + 2.
            Default is None.
        allow_incomplete : bool, optional
            If the polynomial approximation should be performed for features or
            models with incomplete evaluations.
            Default is True.

        Returns
        -------
        U_hat : dict
            A dictionary containing the polynomial approximations for the
            model and each feature as chaospy.Poly objects.
        distribution : chaospy.Dist
            The multivariate distribution for the uncertain parameters.
        data : Data
            A data object containing the values from the model evaluation
            and feature calculations.

        Raises
        ------
        ValueError
            If a common multivariate distribution is given in
            Parameters.distribution and not all uncertain parameters are used.

        Notes
        -----
        The returned `data` should contain (but not necessarily) the following:

            1. ``data["model/features"].evaluations``
            2. ``data["model/features"].time``
            3. ``data["model/features"].labels``
            4. ``data.model_name``
            5. ``data.incomplete``
            6. ``data.method``
            7. ``data.errored``

        The model and feature do not necessarily give results for each
        node. The collocation method is robust towards missing values as long as
        the number of results that remain is high enough.

        The polynomial chaos expansion method for uncertainty quantification
        approximates the model with a polynomial that follows specific
        requirements. This polynomial can be used to quickly calculate the
        uncertainty and sensitivity of the model.

        To create the polynomial chaos expansion we first find the polynomials
        using the three-therm recurrence relation if available, otherwise the
        discretized Stieltjes method is used. Then we use point collocation
        to find the expansion coefficients for the model and each feature of the
        model.

        In point collocation we require the polynomial approximation to be equal
        the model at a set of collocation nodes. This results in a set of linear
        equations for the polynomial coefficients we can solve. We choose
        `nr_collocation_nodes` collocation nodes with Hammersley sampling from
        the `distribution`. We evaluate the model and each feature in parallel,
        and solve the resulting set of linear equations with Tikhonov
        regularization.

        See also
        --------
        uncertainpy.Data
        uncertainpy.Parameters
        """

        uncertain_parameters, distribution, P, nodes, _ = self.get_UQ_samples("collocation", uncertain_parameters, polynomial_order, quadrature_order=nr_collocation_nodes) 

        # Running the model
        data = self.runmodel.run(nodes, uncertain_parameters)

        # TODO: import data here instead of running

        data.method = "polynomial chaos expansion with point collocation. polynomial_order={}, nr_collocation_nodes={}".format(polynomial_order, nr_collocation_nodes)

        logger = get_logger(self)

        U_hat = {}
        # Calculate PC for each feature
        for feature in tqdm(data,
                            desc="Calculating PC for each feature",
                            total=len(data)):
            if feature == self.model.name and self.model.ignore:
                continue

            masked_evaluations, mask, masked_nodes = self.create_masked_nodes(data, feature, nodes)

            if (np.all(mask) or allow_incomplete) and sum(mask) > 0:
                U_hat[feature] = cp.fit_regression(P, masked_nodes,
                                                   masked_evaluations)
            elif not allow_incomplete:
                logger.warning("{}: not all parameter combinations give results.".format(feature) +
                               " No uncertainty quantification is performed since allow_incomplete=False")

            else:
                logger.warning("{}: not all parameter combinations give results.".format(feature))


            if not np.all(mask):
                data.incomplete.append(feature)

        return U_hat, distribution, data


    def create_PCE_spectral_rosenblatt(self,
                                       uncertain_parameters=None,
                                       polynomial_order=4,
                                       quadrature_order=None,
                                       allow_incomplete=True):
        """
        Create the polynomial approximation `U_hat` using pseudo-spectral
        projection and the Rosenblatt transformation. Works for dependend
        uncertain parameters.

        Parameters
        ----------
        uncertain_parameters : {None, str, list}, optional
            The uncertain parameter(s) to use when creating the polynomial
            approximation. If None, all uncertain parameters are used.
            Default is None.
        polynomial_order : int, optional
            The polynomial order of the polynomial approximation.
            Default is 4.
        quadrature_order : {int, None}, optional
            The order of the Leja quadrature method. If None,
            ``quadrature_order = polynomial_order + 2``.
            Default is None.
        allow_incomplete : bool, optional
            If the polynomial approximation should be performed for features or
            models with incomplete evaluations.
            Default is True.

        Returns
        -------
        U_hat : dict
            A dictionary containing the polynomial approximations for the
            model and each feature as chaospy.Poly objects.
        distribution : chaospy.Dist
            The multivariate distribution for the uncertain parameters.
        data : Data
            A data object containing the values from the model evaluation
            and feature calculations.

        Raises
        ------
        ValueError
            If a common multivariate distribution is given in
            Parameters.distribution and not all uncertain parameters are used.

        Notes
        -----
        `data` should contain (but not necessarily) the following, if
           applicable:

            1. ``data["model/features"].evaluations``
            2. ``data["model/features"].time``
            3. ``data["model/features"].labels``
            4. ``data.model_name``
            5. ``data.incomplete``
            6. ``data.method``
            7. ``data.errored``

        The model and feature do not necessarily give results for each
        node. The pseudo-spectral methods is sensitive to missing values, so
        `allow_incomplete` should be used with care.

        The polynomial chaos expansion method for uncertainty quantification
        approximates the model with a polynomial that follows specific
        requirements. This polynomial can be used to quickly calculate the
        uncertainty and sensitivity of the model.

        We use the Rosenblatt transformation to transform from dependent to
        independent variables before we create the polynomial chaos expansion.
        We first find the polynomials from the independent distributions
        using the three-therm recurrence relation if available, otherwise the
        discretized Stieltjes method is used. Then we use the pseudo-spectral
        projection with the Rosenblatt transformation to find the expansion
        coefficients for the model and each feature of the model.

        Pseudo-spectral projection is based on least squares
        minimization and finds the expansion coefficients through numerical
        integration. The integration uses a quadrature scheme with weights
        and nodes. We use Leja quadrature with Smolyak sparse grids to reduce the
        number of nodes required.
        We use the Rosenblatt transformation to transform the quadrature nodes
        before they are sent to the model evaluation.
        For each of the nodes we evaluate the model and calculate the features,
        and the polynomial approximation is created from these results.

        See also
        --------
        uncertainpy.Data
        uncertainpy.Parameters
        """

        uncertain_parameters = self.convert_uncertain_parameters(uncertain_parameters)

        distribution = self.create_distribution(uncertain_parameters=uncertain_parameters)


        # Create the Multivariate normal distribution
        dist_R = []
        for parameter in uncertain_parameters:
            dist_R.append(cp.Normal())

        dist_R = cp.J(*dist_R)

        P = cp.orth_ttr(polynomial_order, dist_R)

        if quadrature_order is None:
            quadrature_order = polynomial_order + 2

        nodes_R, weights_R = cp.generate_quadrature(quadrature_order,
                                                    dist_R,
                                                    rule="J",
                                                    sparse=True)


        nodes = distribution.inv(dist_R.fwd(nodes_R))
        # weights = weights_R*distribution.pdf(nodes)/dist_R.pdf(nodes_R)

        # Running the model
        data = self.runmodel.run(nodes, uncertain_parameters)

        data.method = "polynomial chaos expansion with the pseudo-spectral method and the Rosenblatt transformation. polynomial_order={}, quadrature_order={}".format(polynomial_order, quadrature_order)

        logger = get_logger(self)

        U_hat = {}
        # Calculate PC for each feature
        for feature in tqdm(data,
                            desc="Calculating PC for each feature",
                            total=len(data)):

            if feature == self.model.name and self.model.ignore:
                continue

            # The tutorial version
            # masked_nodes, masked_values, mask, masked_weights = self.create_mask(data,
            #                                                           nodes_R,
            #                                                           feature,
            #                                                           weights)

            # The version thats seems to be working
            masked_evaluations, mask, masked_nodes, masked_weights = \
                self.create_masked_nodes_weights(data,
                                                 feature,
                                                 nodes_R,
                                                 weights_R)

            if (np.all(mask) or allow_incomplete) and sum(mask) > 0:
                U_hat[feature] = cp.fit_quadrature(P,
                                                   masked_nodes,
                                                   masked_weights,
                                                   masked_evaluations)
            elif not allow_incomplete:
                logger.warning("{}: not all parameter combinations give results.".format(feature) +
                               " No uncertainty quantification is performed since allow_incomplete=False")

            else:
                logger.warning("{}: not all parameter combinations give results.".format(feature))


            if not np.all(mask):
                data.incomplete.append(feature)

        return U_hat, dist_R, data


    def create_PCE_collocation_rosenblatt(self,
                                          uncertain_parameters=None,
                                          polynomial_order=4,
                                          nr_collocation_nodes=None,
                                          allow_incomplete=True):
        """
        Create the polynomial approximation `U_hat` using pseudo-spectral
        projection and the Rosenblatt transformation. Works for dependend
        uncertain parameters.

        Parameters
        ----------
        uncertain_parameters : {None, str, list}, optional
            The uncertain parameter(s) to use when creating the polynomial
            approximation. If None, all uncertain parameters are used.
            Default is None.
        polynomial_order : int, optional
            The polynomial order of the polynomial approximation.
            Default is 4.
        nr_collocation_nodes : {int, None}, optional
            The number of collocation nodes to choose. If None,
            `nr_collocation_nodes` = 2* number of expansion factors + 2.
            Default is None.
        allow_incomplete : bool, optional
            If the polynomial approximation should be performed for features or
            models with incomplete evaluations.
            Default is True.

        Returns
        -------
        U_hat : dict
            A dictionary containing the polynomial approximations for the
            model and each feature as chaospy.Poly objects.
        distribution : chaospy.Dist
            The multivariate distribution for the uncertain parameters.
        data : Data
            A data object containing the values from the model evaluation
            and feature calculations.

        Raises
        ------
        ValueError
            If a common multivariate distribution is given in
            Parameters.distribution and not all uncertain parameters are used.

        Notes
        -----
        The returned `data` should contain (but not necessarily) the following:

            1. ``data["model/features"].evaluations``
            2. ``data["model/features"].time``
            3. ``data["model/features"].labels``
            4. ``data.model_name``
            5. ``data.incomplete``
            6. ``data.method``

        The model and feature do not necessarily give results for each node. The
        collocation method is robust towards missing values as long as the number
        of results that remain is high enough.

        The polynomial chaos expansion method for uncertainty quantification
        approximates the model with a polynomial that follows specific
        requirements. This polynomial can be used to quickly calculate the
        uncertainty and sensitivity of the model.

        We use the Rosenblatt transformation to transform from dependent to
        independent variables before we create the polynomial chaos expansion.
        We first find the polynomials from the independent distributions using
        the three-therm recurrence relation if available, otherwise the
        discretized Stieltjes method is used. Then we use the point collocation
        with the Rosenblatt transformation to find the expansion coefficients
        for the model and each feature of the model.

        In point collocation we require the polynomial approximation to be equal
        the model at a set of collocation nodes. This results in a set of linear
        equations for the polynomial coefficients we can solve. We choose
        `nr_collocation_nodes` collocation nodes with Hammersley sampling from
        the independent distribution. We then transform the nodes using the
        Rosenblatte transformation and evaluate the model and each
        feature in parallel. We solve the resulting set of linear equations
        with Tikhonov regularization.

        See also
        --------
        uncertainpy.Data
        uncertainpy.Parameters
        """
        uncertain_parameters = self.convert_uncertain_parameters(uncertain_parameters)

        distribution = self.create_distribution(uncertain_parameters=uncertain_parameters)


        # Create the Multivariate normal distribution
        # dist_R = cp.Iid(cp.Normal(), len(uncertain_parameters))
        dist_R = []
        for parameter in uncertain_parameters:
            dist_R.append(cp.Normal())

        dist_R = cp.J(*dist_R)

        P = cp.orth_ttr(polynomial_order, dist_R)

        if nr_collocation_nodes is None:
            nr_collocation_nodes = 2*len(P) + 2

        nodes_R = dist_R.sample(nr_collocation_nodes, "M")
        nodes = distribution.inv(dist_R.fwd(nodes_R))

        # Running the model
        data = self.runmodel.run(nodes, uncertain_parameters)

        data.method = "polynomial chaos expansion with point collocation and the Rosenblatt transformation. polynomial_order={}, nr_collocation_nodes={}".format(polynomial_order, nr_collocation_nodes)

        logger = get_logger(self)

        U_hat = {}
        # Calculate PC for each feature
        for feature in tqdm(data,
                            desc="Calculating PC for each feature",
                            total=len(data)):
            if feature == self.model.name and self.model.ignore:
                continue

            masked_evaluations, mask, masked_nodes = self.create_masked_nodes(data, feature, nodes_R)

            if (np.all(mask) or allow_incomplete) and sum(mask) > 0:
                U_hat[feature] = cp.fit_regression(P,
                                                   masked_nodes,
                                                   masked_evaluations)
            elif not allow_incomplete:
                logger.warning("{}: not all parameter combinations give results.".format(feature) +
                               " No uncertainty quantification is performed since allow_incomplete=False")

            else:
                logger.warning("{}: not all parameter combinations give results.".format(feature))

            if not np.all(mask):
                data.incomplete.append(feature)

        return U_hat, dist_R, data


    def analyse_PCE(self, U_hat, distribution, data, nr_samples=10**4, save_samples=False):
        """
        Calculate the statistical metrics from the polynomial chaos
        approximation.

        Parameters
        ----------
        U_hat : dict
            A dictionary containing the polynomial approximations for the
            model and each feature as chaospy.Poly objects.
        distribution : chaospy.Dist
            The multivariate distribution for the uncertain parameters.
        data : Data
            A data object containing the values from the model evaluation
            and feature calculations.
        nr_samples : int, optional
            Number of samples for the Monte Carlo sampling of the polynomial
            chaos approximation.
            Default is 10**4.
        save_samples: bool, optional
            Save samples in feature data to, for example, plot PDFs later.

        Returns
        -------
        data : Data
            The `data` parameter given as input with the statistical metrics added.

        Notes
        -----
        The `data` parameter should contain (but not necessarily) the following:

            1. ``data["model/features"].evaluations``
            2. ``data["model/features"].time``
            3. ``data["model/features"].labels``
            4. ``data.model_name``
            5. ``data.incomplete``
            6. ``data.method``
            7. ``data.errored``

        When returned `data` additionally contains:

            8. ``data["model/features"].mean``
            9. ``data["model/features"].variance``
            10. ``data["model/features"].percentile_5``
            11. ``data["model/features"].percentile_95``
            12. ``data["model/features"].sobol_first``, if more than 1 parameter
            13. ``data["model/features"].sobol_total``, if more than 1 parameter
            14. ``data["model/features"].sobol_first_average``, if more than 1 parameter
            15. ``data["model/features"].sobol_total_average``, if more than 1 parameter

        See also
        --------
        uncertainpy.Data
        """

        if len(data.uncertain_parameters) == 1:
            logger = get_logger(self)
            logger.info("Only 1 uncertain parameter. Sensitivities are not calculated")

        U_mc = {}
        for feature in tqdm(data,
                            desc="Calculating statistics from PCE",
                            total=len(data)):
            if feature in U_hat:
                data[feature].mean = cp.E(U_hat[feature], distribution)
                data[feature].variance = cp.Var(U_hat[feature], distribution)

                samples = distribution.sample(nr_samples, "M")

                if len(data.uncertain_parameters) > 1:
                    U_mc[feature] = U_hat[feature](*samples)

                    data[feature].sobol_first = cp.Sens_m(U_hat[feature], distribution)
                    data[feature].sobol_total = cp.Sens_t(U_hat[feature], distribution)
                    data = self.average_sensitivity(data, sensitivity="sobol_first")
                    data = self.average_sensitivity(data, sensitivity="sobol_total")

                else:
                    U_mc[feature] = U_hat[feature](samples)

                if save_samples:
                    data[feature].samples = U_mc[feature]
                data[feature].percentile_5 = np.percentile(U_mc[feature], 5, -1)
                data[feature].percentile_95 = np.percentile(U_mc[feature], 95, -1)

        return data



    @property
    def create_PCE_custom(self, uncertain_parameters=None, **kwargs):
        """
        A custom method for calculating the polynomial chaos approximation.
        Must follow the below requirements.

        Parameters
        ----------
        self : UncertaintyCalculation
            An explicit self is required as the first argument.
            self can be used inside the custom function.
        uncertain_parameters : {None, str, list}, optional
            The uncertain parameter(s) to use when creating the polynomial
            approximation. If None, all uncertain parameters are used.
            Default is None.
        **kwargs
            Any number of optional arguments.

        Returns
        -------
        U_hat : dict
            A dictionary containing the polynomial approximations for the
            model and each feature as chaospy.Poly objects.
        distribution : chaospy.Dist
            The multivariate distribution for the uncertain parameters.
        data : Data
            A data object containing the values from the model evaluation
            and feature calculations.

        Raises
        ------
        ValueError
            If a common multivariate distribution is given in
            Parameters.distribution and not all uncertain parameters are used.

        Notes
        -----
        This method can be implemented to create a custom method to calculate
        the polynomial chaos expansion. The method must calculate and return
        the return arguments described above.

        The returned `data` should contain (but not necessarily) the following:

            1. ``data["model/features"].evaluations``
            2. ``data["model/features"].time``
            3. ``data["model/features"].labels``
            4. ``data.model_name``
            5. ``data.incomplete``
            6. ``data.method``

        The method `analyse_PCE` is called after the polynomial approximation
        has been created.

        Usefull methods in Uncertainpy are:

        1. uncertainpy.core.Uncertaintycalculations.convert_uncertain_parameters
        2. uncertainpy.core.Uncertaintycalculations.create_distribution
        3. uncertainpy.core.RunModel.run

        See also
        --------
        uncertainpy.Data
        uncertainpy.Parameters
        uncertainpy.core.Uncertaintycalculations.convert_uncertain_parameters : Converts uncertain parameters to allowed list
        uncertainpy.core.Uncertaintycalculations.create_distribution : Creates the uncertain parameter distribution
        uncertainpy.core.RunModel.run : Runs the model
        """
        return self._create_PCE_custom

    @create_PCE_custom.setter
    def create_PCE_custom(self, new_create_PCE_custom):
        if not callable(new_create_PCE_custom):
            raise TypeError("create_PCE_custom function must be callable")

        self._create_PCE_custom = types.MethodType(new_create_PCE_custom, self)
        # self._create_PCE_custom = new_create_PCE_custom


    def _create_PCE_custom(self, uncertain_parameters=None, **kwargs):
        raise NotImplementedError("No custom Polynomial Chaos expansion method implemented")


    @property
    def custom_uncertainty_quantification(self, **kwargs):
        """
        A custom uncertainty quantification method. Must follow the below
        requirements.

        Parameters
        ----------
        self : UncertaintyCalculation
            An explicit self is required as the first argument.
            self can be used inside the custom function.
        **kwargs
            Any number of optional arguments.

        Returns
        -------
        data : Data
            A Data object with calculated uncertainties.

        Notes
        -----
        Usefull methods in Uncertainpy are:

        1. uncertainpy.core.Uncertaintycalculations.convert_uncertain_parameters
           - Converts uncertain parameters to an allowed list.
        2. uncertainpy.core.Uncertaintycalculations.create_distribution
           - Creates the uncertain parameter distribution
        3. uncertainpy.core.RunModel.run - Runs the model and all features.

        See also
        --------
        uncertainpy.Data
        uncertainpy.core.Uncertaintycalculations.convert_uncertain_parameters : Converts uncertain parameters to list
        uncertainpy.core.Uncertaintycalculations.create_distribution : Create uncertain parameter distribution
        uncertainpy.core.RunModel.run : Runs the model
        """

        return self._custom_uncertainty_quantification

    @custom_uncertainty_quantification.setter
    def custom_uncertainty_quantification(self, new_custom_uncertainty_quantification):
        if not callable(new_custom_uncertainty_quantification):
            raise TypeError("custom_uncertainty_quantification function must be callable")

        self._custom_uncertainty_quantification = types.MethodType(new_custom_uncertainty_quantification, self)
        # self._custom_uncertainty_quantification = new_custom_uncertainty_quantification

    def _custom_uncertainty_quantification(self, **kwargs):
        raise NotImplementedError("No custom uncertainty calculation method implemented")


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
                         save_samples=False,
                         **custom_kwargs):
        """
        Perform an uncertainty quantification and sensitivity analysis
        using polynomial chaos expansions.

        Parameters
        ----------
        method : {"collocation", "spectral", "custom"}, optional
            The method to use when creating the polynomial chaos approximation.
            "collocation" is the point collocation method "spectral" is
            pseudo-spectral projection, and "custom" is the custom polynomial
            method.
            Default is "collocation".
        rosenblatt : {"auto", bool}, optional
            If the Rosenblatt transformation should be used. The Rosenblatt
            transformation must be used if the uncertain parameters have
            dependent variables. If "auto" the Rosenblatt transformation is used
            if there are dependent parameters, and it is not used of the
            parameters have independent distributions. Default is "auto".
        uncertain_parameters : {None, str, list}, optional
            The uncertain parameter(s) to use when creating the polynomial
            approximation. If None, all uncertain parameters are used.
            Default is None.
        polynomial_order : int, optional
            The polynomial order of the polynomial approximation.
            Default is 4.
        nr_collocation_nodes : {int, None}, optional
            The number of collocation nodes to choose, if point collocation is
            used. If None, `nr_collocation_nodes` = 2* number of expansion factors + 2.
            Default is None.
        quadrature_order : {int, None}, optional
            The order of the Leja quadrature method, if pseudo-spectral
            projection is used. If None, ``quadrature_order = polynomial_order + 2``.
            Default is None.
        nr_pc_mc_samples : int, optional
            Number of samples for the Monte Carlo sampling of the polynomial
            chaos approximation.
        allow_incomplete : bool, optional
            If the polynomial approximation should be performed for features or
            models with incomplete evaluations.
            Default is True.
        seed : int, optional
            Set a random seed. If None, no seed is set. Default is None.

        Returns
        -------
        data : Data
            A data object with all model and feature values, as well as all
            calculated statistical metrics.

        Raises
        ------
        ValueError
            If a common multivariate distribution is given in
            Parameters.distribution and not all uncertain parameters are used.
        ValueError
            If `method` not one of "collocation", "spectral" or "custom".
        NotImplementedError
            If "custom" is chosen and have not been implemented.

        Notes
        -----
        The returned `data` should contain the following:

            1. ``data["model/features"].evaluations``
            2. ``data["model/features"].time``
            3. ``data["model/features"].labels``
            4. ``data.model_name``
            5. ``data.incomplete``
            6. ``data.method``
            7. ``data.errored``
            8. ``data["model/features"].mean``
            9. ``data["model/features"].variance``
            10. ``data["model/features"].percentile_5``
            11. ``data["model/features"].percentile_95``
            12. ``data["model/features"].sobol_first``, if more than 1 parameter
            13. ``data["model/features"].sobol_total``, if more than 1 parameter
            14. ``data["model/features"].sobol_first_average``, if more than 1 parameter
            15. ``data["model/features"].sobol_total_average``, if more than 1 parameter

        The model and feature do not necessarily give results for each
        node. The collocation method is robust towards missing values as long as
        the number of results that remain is high enough. The pseudo-spectral
        method on the other hand, is sensitive to missing values, so
        `allow_incomplete` should be used with care in that case.

        The polynomial chaos expansion method for uncertainty quantification
        approximates the model with a polynomial that follows specific
        requirements. This polynomial can be used to quickly calculate the
        uncertainty and sensitivity of the model.

        To create the polynomial chaos expansion we first find the polynomials
        using the three-therm recurrence relation if available,
        otherwise the discretized Stieltjes method is used. Then we use point collocation
        or pseudo-spectral projection to find the expansion coefficients for the
        model and each feature of the model.

        In point collocation we require the polynomial approximation to be equal
        the model at a set of collocation nodes. This results in a set of linear
        equations for the polynomial coefficients we can solve. We choose
        `nr_collocation_nodes` collocation nodes with Hammersley sampling from
        the `distribution`. We evaluate the model and each feature in parallel,
        and solve the resulting set of linear equations with Tikhonov
        regularization.

        Pseudo-spectral projection is based on least squares minimization and
        finds the expansion coefficients through numerical integration. The
        integration uses a quadrature scheme with weights and nodes. We use Leja
        quadrature with Smolyak sparse grids to reduce the number of nodes
        required. For each of the nodes we evaluate the model and calculate the
        features, and the polynomial approximation is created from these results.

        If we have dependent uncertain parameters we must use the Rosenblatt
        transformation. We use the Rosenblatt transformation to transform from
        dependent to independent variables before we create the polynomial chaos
        expansion. We first find the polynomials from the independent
        distributions using the three-term recurrence relation if available,
        otherwise the discretized Stieltjes method is used

        Both pseudo-spectral projection and point collocation is performed using
        the independent distribution, the only difference is that we use the
        Rosenblatt transformation to transform the nodes from the independent
        distribution to the dependent distribution.

        See also
        --------
        uncertainpy.Data
        uncertainpy.Parameters
        """
        if seed is not None:
            np.random.seed(seed)

        uncertain_parameters = self.convert_uncertain_parameters(uncertain_parameters)
        distribution = self.create_distribution(uncertain_parameters=uncertain_parameters)

        if rosenblatt == "auto":
            if self.dependent(distribution):
                rosenblatt = True
            else:
                rosenblatt = False

        elif rosenblatt == False:
            if self.dependent(distribution):
                raise ValueError('Dependent parameters require using the Rosenblatt transformation. Set rosenblatt="auto" or rosenblatt=True')

        if method == "collocation":
            if rosenblatt:
                U_hat, distribution, data = \
                    self.create_PCE_collocation_rosenblatt(uncertain_parameters=uncertain_parameters,
                                                           polynomial_order=polynomial_order,
                                                           nr_collocation_nodes=nr_collocation_nodes,
                                                           allow_incomplete=allow_incomplete)
            else:
                U_hat, distribution, data = \
                    self.create_PCE_collocation(uncertain_parameters=uncertain_parameters,
                                                polynomial_order=polynomial_order,
                                                nr_collocation_nodes=nr_collocation_nodes,
                                                allow_incomplete=allow_incomplete)

        elif method == "spectral":
            if rosenblatt:
                U_hat, distribution, data = \
                    self.create_PCE_spectral_rosenblatt(uncertain_parameters=uncertain_parameters,
                                                        polynomial_order=polynomial_order,
                                                        quadrature_order=quadrature_order,
                                                        allow_incomplete=allow_incomplete)
            else:
                U_hat, distribution, data = \
                    self.create_PCE_spectral(uncertain_parameters=uncertain_parameters,
                                             polynomial_order=polynomial_order,
                                             quadrature_order=quadrature_order,
                                             allow_incomplete=allow_incomplete)

        elif method == "custom":
            U_hat, distribution, data = \
                self.create_PCE_custom(uncertain_parameters, **custom_kwargs)

        # TODO add support for more methods here by using
        # try:
        #     getattr(self, method)
        # except AttributeError:
        #     raise NotImplementedError("{} not implemented".format{method})

        else:
            raise ValueError("No polynomial chaos method with name {}".format(method))

        data = self.analyse_PCE(U_hat, distribution, data,
                                nr_samples=nr_pc_mc_samples,
                                save_samples=save_samples)

        data.seed = seed

        return data

    def monte_carlo(self,
                    uncertain_parameters=None,
                    nr_samples=10**4,
                    seed=None,
                    allow_incomplete=True,
                    save_samples=False):
        """
        Perform an uncertainty quantification using the quasi-Monte Carlo method.

        Parameters
        ----------
        uncertain_parameters : {None, str, list}, optional
            The uncertain parameter(s) to use when creating the polynomial
            approximation. If None, all uncertain parameters are used.
            Default is None.
        nr_samples : int, optional
            Number of samples for the quasi-Monte Carlo sampling.
            Default is 10**4.
        seed : int, optional
            Set a random seed. If None, no seed is set.
            Default is None.
        allow_incomplete : bool, optional
            If the uncertainty quantification should be performed for features
            or models with incomplete evaluations.
            Default is True.
        save_samples: bool, optional
            Save samples in feature data to, for example, plot PDFs later.

        Returns
        -------
        data : Data
            A data object with all model and feature evaluations, as well as all
            calculated statistical metrics.

        Raises
        ------
        ValueError
            If a common multivariate distribution is given in
            Parameters.distribution and not all uncertain parameters are used.

        Notes
        -----
        The returned `data` should contain the following:

            1. ``data["model/features"].evaluations``
            2. ``data["model/features"].time``
            3. ``data["model/features"].labels``
            4. ``data.model_name``
            5. ``data.incomplete``
            6. ``data.method``
            7. ``data.errored``
            8. ``data["model/features"].mean``
            9. ``data["model/features"].variance``
            10. ``data["model/features"].percentile_5``
            11. ``data["model/features"].percentile_95``
            12. ``data["model/features"].sobol_first``, if more than 1 parameter
            13. ``data["model/features"].sobol_total``, if more than 1 parameter
            14. ``data["model/features"].sobol_first_average``, if more than 1 parameter
            15. ``data["model/features"].sobol_total_average``, if more than 1 parameter


        In the quasi-Monte Carlo method we quasi-randomly draw
        ``(nr_samples/2)*(nr_uncertain_parameters + 2)`` (nr_samples=10**4 by default)
        parameter samples using Saltelli's sampling scheme ([1]_). We require
        this number of samples to be able to calculate the Sobol indices. We
        evaluate the model for each of these parameter samples and calculate the
        features from each of the model results. This step is performed in
        parallel to speed up the calculations. Then we use nr_samples` of
        the model and feature results to calculate the mean, variance, and 5th
        and 95th percentile for the model and each feature. Lastly, we use all
        calculated model and each feature results to calculate the Sobol indices
        using Saltellie's approach.

        References
        ----------
        .. [1] Saltelli, A., P. Annoni, I. Azzini, F. Campolongo, M. Ratto, and
            S. Tarantola (2010).  "Variance based sensitivity analysis of model
            output.  Design and estimator for the total sensitivity index."
            Computer Physics Communications, 181(2):259-270,
            doi:10.1016/j.cpc.2009.09.018.

        See also
        --------
        uncertainpy.Data
        uncertainpy.Parameters
        """

        uncertain_parameters, distribution, _, nodes, nr_sobol_samples = self.get_UQ_samples("MC", uncertain_parameters, nr_samples)

        data = self.runmodel.run(nodes, uncertain_parameters)

        data.method = "monte carlo method. nr_samples={}".format(nr_samples)
        data.seed = seed

        logger = get_logger(self)
        for feature in data:
            if feature == self.model.name and self.model.ignore:
                continue

            # Only use A to calculate the mean and variance
            A, B, AB = self.separate_output_values(data[feature].evaluations,
                                                   len(uncertain_parameters),
                                                   nr_sobol_samples)

            independent_evaluations = np.concatenate([A, B])

            masked_evaluations, mask = self.create_mask(independent_evaluations)

            logger = get_logger(self)

            if (np.all(mask) or allow_incomplete) and sum(mask) > 0:
                if save_samples:
                    data[feature].samples = masked_evaluations
                data[feature].mean = np.mean(masked_evaluations, 0)
                data[feature].variance = np.var(masked_evaluations, 0)

                data[feature].percentile_5 = np.percentile(masked_evaluations, 5, 0)
                data[feature].percentile_95 = np.percentile(masked_evaluations, 95, 0)

                if len(data.uncertain_parameters) > 1:
                    # Results cannot be removed when calculating the sensitivity.
                    # Instead NaN results are set to the mean.
                    # see https://github.com/SALib/SALib/issues/134
                    _, mask = self.create_mask(data[feature].evaluations)
                    masked_mean_evaluations = data[feature].evaluations

                    # masked_mean_evaluations[~mask] = data[feature].mean
                    indices = np.where(mask == 0)[0]

                    for i in indices:
                        masked_mean_evaluations[i] = data[feature].mean

                    if not np.all(mask):
                        logger.warning("{}: only yields ".format(feature) +
                                       "results for {}/{} ".format(sum(mask), len(mask)) +
                                       "parameter combinations." +
                                       "numpy.nan results are set to the mean when calculating the Sobol indices. " +
                                       "This might affect the Sobol indices.")


                    sobol_first, sobol_total = self.mc_calculate_sobol(masked_mean_evaluations,
                                                                       len(uncertain_parameters),
                                                                       nr_sobol_samples)
                    data[feature].sobol_first = sobol_first
                    data[feature].sobol_total = sobol_total
                    data = self.average_sensitivity(data, sensitivity="sobol_first")
                    data = self.average_sensitivity(data, sensitivity="sobol_total")

            elif not allow_incomplete:
                logger.warning("{}: not all parameter combinations give results.".format(feature) +
                               " No uncertainty quantification is performed since allow_incomplete=False")

            else:
                logger.warning("{}: not all parameter combinations give results.".format(feature))

            if not np.all(mask):
                data.incomplete.append(feature)


        return data


    def separate_output_values(self, evaluations, nr_uncertain_parameters, nr_samples):
        """
        Notes
        -----
        Separate the output from the model evaluations, evaluated for the
        samples created by SALIB.sample.saltelli.

        Parameters
        ----------
        evaluations : array_like
            The model evaluations, evaluated for the samples created by
            SALIB.sample.saltelli.
        nr_uncertain_parameters : int
            Number of uncertain parameters.
        nr_samples : int
            Number of samples used in the Monte Carlo sampling.

        Returns
        ----------
        A : array_like
            The A sample matrix from saltellie et. al. 2010.
        B : array_like
            The B sample matrix from saltellie et. al. 2010.
        AB : array_like
            The AB sample matrix from saltellie et. al. 2010.

        Notes
        -----
        Adapted from SALib/analyze/sobol.py:

        https://github.com/SALib/SALib/blob/master/SALib/analyze/sobol.py
        """

        evaluations = np.array(evaluations)

        shape = (nr_samples, nr_uncertain_parameters) + evaluations[0].shape
        step = nr_uncertain_parameters + 2
        AB = np.zeros(shape)

        A = evaluations[0:evaluations.shape[0]:step]
        B = evaluations[(step - 1):evaluations.shape[0]:step]

        for i in range(nr_uncertain_parameters):
            AB[:, i] = evaluations[(i + 1):evaluations.shape[0]:step]

        return A, B, AB


    def mc_calculate_sobol(self, evaluations, nr_uncertain_parameters, nr_samples):
        """
        Calculate the Sobol indices.

        Parameters
        ----------
        evaluations : array_like
            The model evaluations, evaluated for the samples created by
            SALIB.sample.saltelli.
        nr_uncertain_parameters : int
            Number of uncertain parameters.
        nr_samples : int
            Number of samples used in the Monte Carlo sampling.

        Returns
        ----------
        sobol_first : list
            The first order Sobol indices for each uncertain parameter.
        sobol_total : list
            The total order Sobol indices for each uncertain parameter.
        """
        sobol_first = [0]*nr_uncertain_parameters
        sobol_total = [0]*nr_uncertain_parameters

        A, B, AB = self.separate_output_values(evaluations, nr_uncertain_parameters, nr_samples)

        for i in range(nr_uncertain_parameters):
            sobol_first[i] = first_order(A, AB[:, i], B)
            sobol_total[i] = total_order(A, AB[:, i], B)

        return sobol_first, sobol_total


    def average_sensitivity(self, data, sensitivity="sobol_first"):
        """
        Calculate the average of the sensitivities for the model and all
        features and add them to `data`. Ignores any occurrences of numpy.NaN.

        Parameters
        ----------
        data : Data
            A data object with all model and feature evaluations, as well as all
            calculated statistical metrics.
        sensitivity : {"sobol_first", "first", "sobol_total", "total"}, optional
            The sensitivity to normalize and sum. "sobol_first" and "1" are
            for the first order Sobol indice while "sobol_total" and "t" is
            for the total order Sobol indices. Default is "sobol_first".

        Returns
        ----------
        data : Data
            The `data` object with the average of the sensitivities for
            the model and all features added.

        See also
        --------
        uncertainpy.Data
        """
        if sensitivity not in ["sobol_first", "first", "sobol_total", "total"]:
            raise ValueError("Sensitivity must be either: sobol_first, first, sobol_total, total, not {}".format(sensitivity))

        if sensitivity == "first":
            sensitivity = "sobol_first"
        elif sensitivity == "total":
            sensitivity = "sobol_total"

        for feature in data:
            if sensitivity in data[feature]:
                total_sense = []
                for i in range(0, len(data.uncertain_parameters)):
                    total_sense.append(np.nanmean(data[feature][sensitivity][i]))

                data[feature][sensitivity + "_average"] = np.array(total_sense)


        return data
