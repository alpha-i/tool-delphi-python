from alphai_prototype_env.controllers.base_controller import BaseController


class Controller(BaseController):

    def __init__(self, baseline):
        super().__init__()
        self.baseline = baseline

    def compare_to_baseline(self, oracle, data_source):

        self.check_model_compatible_with_baseline(oracle)

        self.train_oracle(self.baseline, data_source)
        self.train_oracle(oracle, data_source)

        baseline_metrics = self.test_oracle(self.baseline, data_source)
        model_metrics = self.test_oracle(oracle, data_source)

        results = {"baseline_metrics": baseline_metrics, "model_metrics": model_metrics}

        return results

    def check_model_compatible_with_baseline(self, oracle):

        if oracle.trade_frequency != self.baseline.trade_frequency:
            raise ValueError('oracle prediction frequency should be equal to the baseline prediction frequency')

        if oracle.trade_delta != self.baseline.trade_delta:
            raise ValueError('oracle prediction delta should be equal to the baseline prediction delta')

        if oracle.trade_offset != self.baseline.trade_offset:
            raise ValueError('model prediction offset should be equal to the baseline prediction offset')

    def compare_multiple_oracles(self, oracle_list, data_source):

        results_dict = {}
        for oracle in oracle_list:
            results = self.compare_to_baseline(oracle, data_source)
            results_dict[oracle.__class__.__name__] = results
