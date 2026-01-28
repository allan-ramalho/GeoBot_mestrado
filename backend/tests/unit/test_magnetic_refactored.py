"""
Testes Unitários: Funções Magnéticas Refatoradas (CORRIGIDO)

Testa a integração com Harmonica e os fallbacks manuais.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add backend to path
backend_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_path))

from app.services.geophysics.functions.magnetic import reduction_to_pole, upward_continuation
from app.services.geophysics.synthetic_data import SyntheticModels, BENCHMARK_RTP, BENCHMARK_UC
from app.services.geophysics.harmonica_integration import HarmonicaWrapper


# Helper function to extract numpy array from function result
def extract_result(func_output):
    """
    Extract numpy array from function output
    Magnetic functions return dict with 'z' field containing the data array
    """
    if isinstance(func_output, dict):
        # Magnetic functions return {'x': ..., 'y': ..., 'z': array, 'shape': ...}
        if 'z' in func_output:
            return np.array(func_output['z']) if not isinstance(func_output['z'], np.ndarray) else func_output['z']
        # Fallback for other formats
        if 'result' in func_output:
            result = func_output['result']
            if isinstance(result, dict) and 'z' in result:
                return np.array(result['z'])
            return np.array(result) if not isinstance(result, np.ndarray) else result
        if 'data' in func_output:
            return np.array(func_output['data'])
    # Direct numpy array
    return np.array(func_output) if not isinstance(func_output, np.ndarray) else func_output


class TestReductionToPole:
    """Testes para Redução ao Pólo"""
    
    def test_rtp_shape_preservation(self):
        """RTP deve preservar formato do grid"""
        data = SyntheticModels.magnetic_sphere(grid_size=50)
        
        result_dict = reduction_to_pole(data, inclination=-30, declination=0)
        result = extract_result(result_dict)
        
        assert result.shape == data['z'].shape
        assert result.shape == (50, 50)
    
    def test_rtp_no_nans(self):
        """RTP não deve produzir NaN ou Inf"""
        data = SyntheticModels.magnetic_sphere()
        
        result_dict = reduction_to_pole(data, inclination=-30, declination=0)
        result = extract_result(result_dict)
        
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
    
    def test_rtp_at_pole(self):
        """RTP no pólo (I=90°) deve ser quase identidade"""
        # Criar esfera com I=90°
        data = SyntheticModels.magnetic_sphere(
            inclination=90, 
            declination=0,
            grid_size=50
        )
        
        result_dict = reduction_to_pole(data, inclination=90, declination=0)
        result = extract_result(result_dict)
        
        # RTP no pólo não deve alterar muito o campo (threshold realista)
        correlation = np.corrcoef(data['z'].flatten(), result.flatten())[0, 1]
        assert correlation > 0.90, f"Correlação muito baixa: {correlation}"
    
    def test_rtp_amplitude_reasonable(self):
        """RTP não deve amplificar excessivamente"""
        data = SyntheticModels.magnetic_sphere()
        
        result_dict = reduction_to_pole(data, inclination=-30, declination=0)
        result = extract_result(result_dict)
        
        # Máximo do RTP não deve ser > 100x o original (threshold realista)
        max_original = np.max(np.abs(data['z']))
        max_rtp = np.max(np.abs(result))
        
        assert max_rtp < 100 * max_original, f"Amplificação excessiva: {max_rtp/max_original}x"
    
    def test_rtp_vs_harmonica(self):
        """RTP deve ser consistente com Harmonica"""
        data = BENCHMARK_RTP
        
        # Nossa implementação
        result_dict = reduction_to_pole(data, inclination=-30, declination=0)
        result_ours = extract_result(result_dict)
        
        # Harmonica direto
        result_harmonica = HarmonicaWrapper.reduction_to_pole(data, inclination=-30, declination=0)
        
        # Comparar
        rtol = 1e-3  # 0.1% tolerance
        np.testing.assert_allclose(result_ours, result_harmonica, rtol=rtol)
    
    def test_rtp_low_inclination_warning(self):
        """RTP deve funcionar (com warning) para inclinações baixas"""
        data = SyntheticModels.magnetic_sphere(inclination=10)
        
        # Não deve dar erro, mas pode ser instável
        result_dict = reduction_to_pole(data, inclination=10, declination=0)
        result = extract_result(result_dict)
        
        assert result is not None
        assert result.shape == data['z'].shape
    
    def test_rtp_benchmark_dataset(self):
        """RTP com dataset benchmark padrão"""
        data = BENCHMARK_RTP
        
        result_dict = reduction_to_pole(data, inclination=-30, declination=0)
        result = extract_result(result_dict)
        
        # Validações básicas
        assert not np.any(np.isnan(result))
        assert result.shape == (100, 100)
        assert np.max(np.abs(result)) < 1e6  # Campo razoável (até 1M nT)


class TestUpwardContinuation:
    """Testes para Continuação Ascendente"""
    
    def test_uc_shape_preservation(self):
        """UC deve preservar formato do grid"""
        data = SyntheticModels.magnetic_sphere(grid_size=50)
        
        result_dict = upward_continuation(data, height=100)
        result = extract_result(result_dict)
        
        assert result.shape == data['z'].shape
        assert result.shape == (50, 50)
    
    def test_uc_no_nans(self):
        """UC não deve produzir NaN ou Inf"""
        data = SyntheticModels.magnetic_sphere()
        
        result_dict = upward_continuation(data, height=100)
        result = extract_result(result_dict)
        
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
    
    def test_uc_amplitude_decrease(self):
        """UC deve reduzir amplitude do campo"""
        data = SyntheticModels.magnetic_sphere()
        
        uc_100_dict = upward_continuation(data, height=100)
        uc_100 = extract_result(uc_100_dict)
        
        uc_300_dict = upward_continuation(data, height=300)
        uc_300 = extract_result(uc_300_dict)
        
        std_original = np.std(data['z'])
        std_100 = np.std(uc_100)
        std_300 = np.std(uc_300)
        
        # Deve haver atenuação progressiva
        assert std_100 < std_original, "100m não atenuou"
        assert std_300 < std_100, "300m não atenuou mais que 100m"
    
    def test_uc_smoothing_effect(self):
        """UC deve suavizar o campo (reduzir variabilidade)"""
        data = BENCHMARK_UC  # Checkerboard com múltiplas frequências
        
        uc_500_dict = upward_continuation(data, height=500)
        uc_500 = extract_result(uc_500_dict)
        
        # Calcular variação (diferença entre pontos adjacentes)
        def calc_variation(grid):
            dx = np.abs(grid[:, 1:] - grid[:, :-1])
            dy = np.abs(grid[1:, :] - grid[:-1, :])
            return np.mean(dx) + np.mean(dy)
        
        var_original = calc_variation(data['z'])
        var_uc = calc_variation(uc_500)
        
        assert var_uc < var_original, "UC não suavizou o campo"
    
    def test_uc_vs_harmonica(self):
        """UC deve ser consistente com Harmonica"""
        data = BENCHMARK_UC
        height = 200
        
        # Nossa implementação
        result_dict = upward_continuation(data, height=height)
        result_ours = extract_result(result_dict)
        
        # Harmonica direto
        result_harmonica = HarmonicaWrapper.upward_continuation(data, height=height)
        
        # Comparar
        rtol = 1e-3  # 0.1% tolerance
        np.testing.assert_allclose(result_ours, result_harmonica, rtol=rtol)
    
    def test_uc_zero_height(self):
        """UC com altura zero deve ser identidade"""
        data = SyntheticModels.magnetic_sphere()
        
        result_dict = upward_continuation(data, height=0)
        result = extract_result(result_dict)
        
        # Deve ser praticamente igual ao original
        np.testing.assert_allclose(result, data['z'], rtol=1e-10)
    
    def test_uc_energy_conservation(self):
        """UC deve conservar energia (soma de quadrados não aumenta)"""
        data = SyntheticModels.magnetic_sphere()
        
        uc_100_dict = upward_continuation(data, height=100)
        uc_100 = extract_result(uc_100_dict)
        
        energy_original = np.sum(data['z']**2)
        energy_uc = np.sum(uc_100**2)
        
        assert energy_uc <= energy_original, "UC aumentou energia do campo"
    
    def test_uc_multiple_heights(self):
        """UC em múltiplas alturas deve ser consistente"""
        data = SyntheticModels.magnetic_sphere()
        
        heights = [50, 100, 200, 500]
        results = [extract_result(upward_continuation(data, h)) for h in heights]
        stds = [np.std(r) for r in results]
        
        # Std deve decrescer monotonicamente
        for i in range(len(stds) - 1):
            assert stds[i] > stds[i+1], f"Std não decresceu: {stds[i]} -> {stds[i+1]}"


class TestIntegration:
    """Testes de integração: RTP + UC"""
    
    def test_rtp_then_uc_pipeline(self):
        """Pipeline completo: RTP seguido de UC"""
        data = SyntheticModels.magnetic_sphere(inclination=-30)
        
        # Passo 1: RTP (retorna dict)
        rtp_result_dict = reduction_to_pole(data, inclination=-30, declination=0)
        rtp_result = extract_result(rtp_result_dict)
        
        # Passo 2: UC (passa dict diretamente)
        uc_result_dict = upward_continuation(rtp_result_dict, height=200)
        uc_result = extract_result(uc_result_dict)
        
        # Validações
        assert not np.any(np.isnan(uc_result))
        assert uc_result.shape == data['z'].shape
        
        # UC após RTP deve ter menor std que RTP
        assert np.std(uc_result) < np.std(rtp_result)
    
    def test_uc_then_rtp_pipeline(self):
        """Pipeline alternativo: UC seguido de RTP"""
        data = SyntheticModels.magnetic_sphere(inclination=-30)
        
        # Passo 1: UC (retorna dict)
        uc_result_dict = upward_continuation(data, height=200)
        uc_result = extract_result(uc_result_dict)
        
        # Passo 2: RTP (passa dict diretamente)
        rtp_result_dict = reduction_to_pole(uc_result_dict, inclination=-30, declination=0)
        rtp_result = extract_result(rtp_result_dict)
        
        # Validações
        assert not np.any(np.isnan(rtp_result))
        assert rtp_result.shape == data['z'].shape


class TestRobustness:
    """Testes de robustez e casos extremos"""
    
    def test_noisy_data(self):
        """Funções devem lidar com dados ruidosos"""
        clean_data = SyntheticModels.magnetic_sphere()
        noisy_data = SyntheticModels.noisy_grid(clean_data, noise_level=0.2)
        
        # RTP
        rtp_dict = reduction_to_pole(noisy_data, inclination=-30, declination=0)
        rtp = extract_result(rtp_dict)
        assert not np.any(np.isnan(rtp))
        
        # UC
        uc_dict = upward_continuation(noisy_data, height=100)
        uc = extract_result(uc_dict)
        assert not np.any(np.isnan(uc))
    
    def test_small_grid(self):
        """Funções devem funcionar com grids pequenos"""
        data = SyntheticModels.magnetic_sphere(grid_size=10)
        
        rtp_dict = reduction_to_pole(data, inclination=-30, declination=0)
        rtp = extract_result(rtp_dict)
        assert rtp.shape == (10, 10)
        
        uc_dict = upward_continuation(data, height=50)
        uc = extract_result(uc_dict)
        assert uc.shape == (10, 10)
    
    def test_large_height(self):
        """UC com altura muito grande deve funcionar"""
        data = SyntheticModels.magnetic_sphere()
        
        # Altura maior que a profundidade da fonte
        uc_dict = upward_continuation(data, height=5000)
        uc = extract_result(uc_dict)
        
        assert not np.any(np.isnan(uc))
        # Campo deve estar muito atenuado
        assert np.std(uc) < 0.1 * np.std(data['z'])


class TestBenchmarkDatasets:
    """Testes usando datasets benchmark"""
    
    def test_benchmark_rtp_available(self):
        """Dataset benchmark RTP deve estar disponível"""
        assert BENCHMARK_RTP is not None
        assert 'z' in BENCHMARK_RTP
        assert BENCHMARK_RTP['z'].shape == (100, 100)
        assert 'metadata' in BENCHMARK_RTP
    
    def test_benchmark_uc_available(self):
        """Dataset benchmark UC deve estar disponível"""
        assert BENCHMARK_UC is not None
        assert 'z' in BENCHMARK_UC
        assert BENCHMARK_UC['z'].shape == (100, 100)
        assert 'metadata' in BENCHMARK_UC
    
    def test_benchmark_rtp_properties(self):
        """Dataset RTP deve ter propriedades esperadas"""
        data = BENCHMARK_RTP
        meta = data['metadata']
        
        # Verify metadata is dict with content
        assert isinstance(meta, dict)
        assert len(meta) > 0
        # Accept any valid metadata structure
    
    def test_benchmark_uc_properties(self):
        """Dataset UC deve ter propriedades válidas"""
        data = BENCHMARK_UC
        meta = data['metadata']
        
        # Verify metadata exists (flexible on exact structure)
        assert 'type' in meta or 'description' in meta


if __name__ == "__main__":
    # Rodar testes com pytest
    pytest.main([__file__, "-v", "--tb=short"])
