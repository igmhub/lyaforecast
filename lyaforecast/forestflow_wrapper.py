# lyaforecast/forestflow_wrapper.py

from lyaforecast.some_module import SomeClass  # Import the necessary FF components


class ForestFlowWrapper:
    """Wrapper around ForestFlow (FF) for integration with lyaforecast."""

    def __init__(self, config):
        """
        Parameters
        ----------
        config : dict
            Configuration options for ForestFlow.
        """
        self.config = config
        self.ff_instance = SomeClass(**config)

    def perform_action(self, data):
        """Process input data using ForestFlow.

        Parameters
        ----------
        data : dict
            Input data to be processed.

        Returns
        -------
        dict
            Processed results from ForestFlow.
        """
        result = self.ff_instance.some_method(data)
        return result

    def custom_method(self, params):
        """Apply custom pre/post-processing logic around a ForestFlow call.

        Parameters
        ----------
        params : dict
            Parameters for the custom operation.

        Returns
        -------
        dict
            Post-processed results.
        """
        preprocessed = self._preprocess(params)
        result = self.ff_instance.some_other_method(preprocessed)
        return self._postprocess(result)

    def _preprocess(self, params):
        """Apply internal preprocessing to input parameters.

        Parameters
        ----------
        params : dict
            Raw input parameters.

        Returns
        -------
        dict
            Preprocessed parameters.
        """
        return params

    def _postprocess(self, result):
        """Apply internal postprocessing to ForestFlow output.

        Parameters
        ----------
        result : dict
            Raw output from ForestFlow.

        Returns
        -------
        dict
            Postprocessed results.
        """
        return result
