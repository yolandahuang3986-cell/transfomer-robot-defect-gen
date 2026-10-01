from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_shared_experiment_plan_and_metric_contract_are_complete():
    experiments = yaml.safe_load((ROOT / 'configs/experiments.yaml').read_text())
    assert [(x['id'], x['training_data']) for x in experiments['experiments']] == [
        ('E0', 'real_only'), ('E1', 'real_plus_procedural'),
        ('E2', 'real_plus_gan'), ('E3', 'real_plus_diffusion')]
    downstream = yaml.safe_load((ROOT / 'configs/downstream.yaml').read_text())
    assert downstream['model']['architecture'] == 'unet'
    assert downstream['metrics']['threshold'] == 0.5
    assert set(downstream['metrics']['names']) == {'pixel_auroc', 'iou', 'dice', 'defect_recall'}
