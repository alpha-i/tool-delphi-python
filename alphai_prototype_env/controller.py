from alphai_prototype_env.metrics import Metrics

from alphai_prototype_env.basic_controller import BasicController


class Controller(BasicController):

    def __init__(self, baseline):
        super().__init__()
        self.baseline = baseline
        self.metrics = Metrics()

    def compare_to_baseline(self, oracle, data_source):

        self.check_model_compatible_with_baseline(oracle)

        self.train_oracle(self.baseline, data_source)
        self.train_oracle(oracle, data_source)

        baseline_metrics = self.test_oracle(self.baseline, data_source)
        model_metrics = self.test_oracle(oracle, data_source)

        results = {"baseline_metrics": baseline_metrics, "model_metrics": model_metrics}

        return results

    def check_model_compatible_with_baseline(self, oracle):

        if oracle.frequency != self.baseline.frequency:
            raise ValueError('model prediction frequency should be equal to the baseline prediction frequency')

        if oracle.delta != self.baseline.delta:
            raise ValueError('model prediction delta should be equal to the baseline prediction delta')

        if oracle.offset != self.baseline.delta:
            raise ValueError('model prediction offset should be equal to the baseline prediction offset')

    def compare_multiple_oracles(self, oracle_list, data_source):

        results_dict = {}
        for oracle in oracle_list:
            results = self.compare_to_baseline(oracle, data_source)
            results_dict[oracle.__class__.__name__] = results
