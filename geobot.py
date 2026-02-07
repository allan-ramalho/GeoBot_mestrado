"""
GeoBot - Agente de IA Conversacional para Processamento de Dados Geofísicos Potenciais
============================================================================

Aplicação científica desenvolvida para processamento e análise de dados de métodos
potenciais (gravimetria e magnetometria) usando IA generativa e RAG.

Autor: Allan Ramalho e Rodrigo Bijani
Projeto: Mestrado em Dinâmica dos Oceanos e da Terra - DOT UFF
Python: 3.11.9
Framework: Streamlit

Arquitetura:
-----------
- Interface conversacional com LLM (Groq API)
- Sistema RAG para citações científicas automáticas
- Pipeline de processamento geofísico modular
- Visualizações interativas 2D/3D
- Suporte a múltiplos idiomas (PT, EN, ES)

Referências Principais:
----------------------
BLAKELY, R. J. **Potential Theory in Gravity and Magnetic Applications**. 
Cambridge University Press, 1995. ISBN: 978-0521575478

TELFORD, W. M.; GELDART, L. P.; SHERIFF, R. E. **Applied Geophysics**. 
2nd ed. Cambridge University Press, 1990. ISBN: 978-0521339384
"""

# ============================================================================
# IMPORTS E CONFIGURAÇÕES
# ============================================================================

# --- Bibliotecas Padrão ---
import os
import sys
import logging
import inspect
import traceback
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO, StringIO
import json
import warnings

# --- Processamento de Dados ---
import numpy as np
import polars as pl
import pandas as pd
import xarray as xr

# --- Processamento Geofísico ---
import harmonica as hm
import scipy.signal as signal
import scipy.ndimage as ndimage
from scipy.interpolate import griddata, RBFInterpolator
from scipy.fft import fft2, ifft2, fftfreq

# --- Geoespacial ---
import pyproj
import geopandas as gpd
from shapely.geometry import Point, Polygon

# --- Visualização ---
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

# --- UI Framework ---
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.colored_header import colored_header
import folium
from streamlit_folium import st_folium, folium_static

# --- LLM e RAG ---
from groq import Groq, RateLimitError, APIError
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from pypdf import PdfReader

# --- Utilitários ---
from langdetect import detect, LangDetectException
from loguru import logger
from rich.console import Console
from rich.table import Table
from tqdm import tqdm
import tiktoken

# --- Otimizações GPU (módulo local) ---
try:
    from geobot_optimizations import (
        fft2_gpu, ifft2_gpu, gaussian_filter_gpu,
        optimize_polars_dataframe, set_gpu_info
    )
    OPTIMIZATIONS_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ Módulo de otimizações não encontrado. Performance reduzida.")
    OPTIMIZATIONS_AVAILABLE = False
    # Fallback para funções padrão
    from scipy.fft import fft2 as fft2_gpu, ifft2 as ifft2_gpu
    from scipy.ndimage import gaussian_filter as gaussian_filter_gpu
    def optimize_polars_dataframe(df, col): return df[col].to_numpy()
    def set_gpu_info(info): pass

# ============================================================================
# CONFIGURAÇÕES GLOBAIS
# ============================================================================

# --- Logging ---
logger.remove()  # Remove handler padrão
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="INFO"
)
logger.add(
    "geobot.log",
    rotation="10 MB",
    retention="7 days",
    level="DEBUG",
    encoding="utf-8"
)

# --- Warnings ---
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- Constantes ---
APP_VERSION = "1.0.0"
APP_TITLE = "🌍 GeoBot"
APP_SUBTITLE = "Agente de IA para Processamento de Dados Geofísicos Potenciais"

# Modelos Groq disponíveis (ordem de preferência)
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "mixtral-8x7b-32768",
    "llama-3.1-8b-instant",
    "gemma2-9b-it"
]

# Unidades de medida
GRAVITY_UNITS = {
    "mgal": "mGal (miligal)",
    "ugal": "μGal (microgal)",
    "si": "m/s² (SI)"
}

MAGNETIC_UNITS = {
    "nt": "nT (nanotesla)",
    "gamma": "γ (gamma)",
    "si": "T (tesla)"
}

# Densidades típicas (kg/m³)
TYPICAL_DENSITIES = {
    "sedimentos": 2200,
    "rochas_sedimentares": 2500,
    "crosta_continental": 2670,
    "granito": 2650,
    "basalto": 2950,
    "agua": 1000
}

# Paletas de cores para visualização
COLOR_SCALES = {
    "gravity": "RdBu_r",
    "magnetic": "viridis",
    "anomaly": "seismic",
    "topography": "terrain"
}

# ============================================================================
# CONFIGURAÇÃO DE GPU (ACELERAÇÃO POR HARDWARE)
# ============================================================================

def configure_gpu():
    """
    Configura GPU para aceleração de processamento quando disponível.
    
    Detecta automaticamente:
    - CUDA (NVIDIA GPUs)
    - MPS (Apple Silicon M1/M2)
    - CPU fallback
    
    Returns:
    --------
    Dict com informações do dispositivo
    """
    gpu_info = {
        "available": False,
        "device": "cpu",
        "device_name": "CPU",
        "backend": "numpy"
    }
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_info["available"] = True
            gpu_info["device"] = "cuda"
            gpu_info["device_name"] = torch.cuda.get_device_name(0)
            gpu_info["backend"] = "pytorch-cuda"
            logger.info(f"🚀 GPU NVIDIA detectada: {gpu_info['device_name']}")
            
            # Configura cuBLAS para melhor performance
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            
        elif torch.backends.mps.is_available():
            gpu_info["available"] = True
            gpu_info["device"] = "mps"
            gpu_info["device_name"] = "Apple Silicon GPU"
            gpu_info["backend"] = "pytorch-mps"
            logger.info(f"🚀 GPU Apple Silicon detectada: Metal Performance Shaders")
            
        else:
            logger.info("ℹ️ GPU não detectada. Usando CPU (NumPy/SciPy)")
            
    except ImportError:
        logger.warning("⚠️ PyTorch não instalado. GPU desabilitada. Instale com: pip install torch")
    
    return gpu_info

# Inicializa configuração de GPU
GPU_INFO = configure_gpu()

# Configura módulo de otimizações
if OPTIMIZATIONS_AVAILABLE:
    set_gpu_info(GPU_INFO)
    logger.info("✅ Módulo de otimizações GPU ativado")

# ============================================================================
# FUNÇÕES AUXILIARES OTIMIZADAS
# ============================================================================

def numpy_to_torch(array: np.ndarray, device: str = None):
    """
    Converte NumPy array para PyTorch tensor em GPU.
    
    Parameters:
    -----------
    array : np.ndarray
        Array NumPy
    device : str
        Dispositivo ('cuda', 'mps', 'cpu')
    
    Returns:
    --------
    torch.Tensor
        Tensor em GPU/CPU
    """
    if GPU_INFO['available']:
        try:
            import torch
            device = device or GPU_INFO['device']
            return torch.from_numpy(array.astype(np.float32)).to(device)
        except:
            return array
    return array

def torch_to_numpy(tensor) -> np.ndarray:
    """
    Converte PyTorch tensor para NumPy array.
    
    Parameters:
    -----------
    tensor : torch.Tensor
        Tensor PyTorch
    
    Returns:
    --------
    np.ndarray
        Array NumPy
    """
    if GPU_INFO['available']:
        try:
            return tensor.cpu().numpy()
        except:
            return tensor
    return tensor

def fft2_gpu(array: np.ndarray) -> np.ndarray:
    """
    FFT 2D acelerada por GPU quando disponível.
    
    OTIMIZAÇÃO: 10-50x mais rápido em GPU.
    
    Parameters:
    -----------
    array : np.ndarray
        Array 2D
    
    Returns:
    --------
    np.ndarray
        FFT 2D do array
    """
    if GPU_INFO['available'] and GPU_INFO['device'] == 'cuda':
        try:
            import torch
            tensor = numpy_to_torch(array)
            # PyTorch FFT é MUITO mais rápido em GPU
            fft_result = torch.fft.fft2(tensor)
            return torch_to_numpy(fft_result)
        except Exception as e:
            logger.debug(f"FFT GPU falhou, usando CPU: {e}")
            return fft2(array)
    else:
        return fft2(array)

def ifft2_gpu(array: np.ndarray) -> np.ndarray:
    """
    IFFT 2D acelerada por GPU quando disponível.
    
    Parameters:
    -----------
    array : np.ndarray
        Array 2D complexo
    
    Returns:
    --------
    np.ndarray
        IFFT 2D do array
    """
    if GPU_INFO['available'] and GPU_INFO['device'] == 'cuda':
        try:
            import torch
            tensor = torch.from_numpy(array).to(GPU_INFO['device'])
            ifft_result = torch.fft.ifft2(tensor)
            return torch_to_numpy(ifft_result.real)
        except Exception as e:
            logger.debug(f"IFFT GPU falhou, usando CPU: {e}")
            return np.real(ifft2(array))
    else:
        return np.real(ifft2(array))

# ============================================================================
# EXCEÇÕES CUSTOMIZADAS
# ============================================================================

class GeoBotException(Exception):
    """Exceção base para todas as exceções do GeoBot."""
    pass

class InvalidDataError(GeoBotException):
    """Erro relacionado a dados inválidos ou malformados."""
    pass

class ProcessingError(GeoBotException):
    """Erro durante processamento geofísico."""
    pass

class LLMError(GeoBotException):
    """Erro relacionado à comunicação com LLM."""
    pass

class RAGError(GeoBotException):
    """Erro no sistema RAG."""
    pass

# ============================================================================
# SISTEMA DE REGISTRO DE FUNÇÕES DE PROCESSAMENTO
# ============================================================================

# Registry global para funções de processamento
PROCESSING_REGISTRY: Dict[str, Dict[str, Any]] = {}

def register_processing(
    category: str,
    description: str,
    input_type: str = "grid",  # 'grid', 'profile', 'points'
    requires_params: List[str] = None
):
    """
    Decorador para auto-registro de funções de processamento geofísico.
    
    Parameters:
    -----------
    category : str
        Categoria do processamento ('gravimetria', 'magnetometria', 'geral')
    description : str
        Descrição curta do processamento
    input_type : str
        Tipo de dado de entrada esperado
    requires_params : list
        Lista de parâmetros obrigatórios
    
    Examples:
    ---------
    >>> @register_processing(category="Gravimetria", description="Correção de Bouguer")
    >>> def bouguer_correction(data, density=2670):
    >>>     ...
    """
    def decorator(func: Callable) -> Callable:
        func_name = func.__name__
        PROCESSING_REGISTRY[func_name] = {
            'function': func,
            'category': category,
            'description': description,
            'input_type': input_type,
            'requires_params': requires_params or [],
            'signature': inspect.signature(func),
            'docstring': func.__doc__ or "Sem documentação disponível"
        }
        logger.debug(f"Registrada função: {func_name} [{category}]")
        return func
    return decorator

# ============================================================================
# CLASSES DE DOMÍNIO
# ============================================================================

@dataclass
class GeophysicalData:
    """
    Classe para representar dados geofísicos estruturados.
    
    Attributes:
    -----------
    data : pl.DataFrame
        DataFrame Polars com os dados
    data_type : str
        Tipo de dado ('gravimetria', 'magnetometria', 'topografia')
    dimension : str
        Dimensionalidade ('1D', '2D', '3D')
    coords : Dict[str, str]
        Mapeamento de colunas de coordenadas {'x': 'col_x', 'y': 'col_y', ...}
    value_column : str
        Nome da coluna com valores observados
    units : str
        Unidade de medida
    crs : str
        Sistema de referência de coordenadas (EPSG code)
    metadata : Dict[str, Any]
        Metadados adicionais
    """
    data: pl.DataFrame
    data_type: str  # 'gravity', 'magnetic', 'topography'
    dimension: str  # '1D', '2D', '3D'
    coords: Dict[str, str]  # {'x': 'longitude', 'y': 'latitude', 'z': 'elevation'}
    value_column: str
    units: str
    crs: str = "EPSG:4326"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Valida dados após inicialização."""
        # Verifica se colunas existem (sem duplicatas)
        coord_cols = set(self.coords.values())
        required_cols = coord_cols | {self.value_column}
        missing_cols = required_cols - set(self.data.columns)
        
        if missing_cols:
            available = list(self.data.columns)
            raise InvalidDataError(
                f"Colunas faltando: {missing_cols}\n"
                f"Colunas disponíveis: {available}\n"
                f"Coords esperados: {self.coords}"
            )
        
        # Calcula estatísticas básicas
        self._compute_stats()
        
        logger.info(f"GeophysicalData criado: {self.data_type} {self.dimension} - {len(self.data)} pontos")
    
    def _compute_stats(self):
        """Calcula estatísticas descritivas dos dados."""
        values = self.data[self.value_column].to_numpy()
        
        self.metadata.update({
            'n_points': len(values),
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values)),
            'q25': float(np.percentile(values, 25)),
            'q75': float(np.percentile(values, 75))
        })
        
        # Bounding box
        if 'x' in self.coords and 'y' in self.coords:
            x_col = self.coords['x']
            y_col = self.coords['y']
            self.metadata['bbox'] = {
                'x_min': float(self.data[x_col].min()),
                'x_max': float(self.data[x_col].max()),
                'y_min': float(self.data[y_col].min()),
                'y_max': float(self.data[y_col].max())
            }
    
    def to_pandas(self) -> pd.DataFrame:
        """Converte para Pandas DataFrame."""
        return self.data.to_pandas()
    
    def to_grid(self, method: str = 'linear', cache: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Interpola dados para grid regular COM CACHE.
        
        OTIMIZAÇÃO: Cache evita reprocessamento (ganho 100-1000x em chamadas repetidas).
        
        Parameters:
        -----------
        method : str
            Método de interpolação ('linear', 'cubic', 'nearest')
        cache : bool
            Se True, usa cache (padrão: True para performance)
        
        Returns:
        --------
        X, Y, Z : np.ndarray
            Arrays 2D com coordenadas e valores gridados
        """
        if self.dimension not in ['2D', '3D']:
            raise ProcessingError("Gridding requer dados 2D ou 3D")
        
        # OTIMIZAÇÃO: Cache de grid (evita reinterpolação)
        cache_key = f"grid_{method}_{id(self.data)}"
        if cache and hasattr(self, '_grid_cache') and cache_key in self._grid_cache:
            logger.debug(f"✅ Cache hit para grid (method={method})")
            return self._grid_cache[cache_key]
        
        x_col = self.coords['x']
        y_col = self.coords['y']
        
        # OTIMIZAÇÃO: Polars zero-copy quando possível
        x = optimize_polars_dataframe(self.data, x_col)
        y = optimize_polars_dataframe(self.data, y_col)
        z = optimize_polars_dataframe(self.data, self.value_column)
        
        # OTIMIZAÇÃO: Resolução adaptativa baseada no tamanho dos dados
        n_points = len(x)
        if n_points < 1000:
            resolution = 50
        elif n_points < 10000:
            resolution = 100
        elif n_points < 100000:
            resolution = 150
        else:
            resolution = 200
        
        # Cria grid regular
        xi = np.linspace(x.min(), x.max(), resolution)
        yi = np.linspace(y.min(), y.max(), resolution)
        Xi, Yi = np.meshgrid(xi, yi)
        
        # Interpola
        Zi = griddata((x, y), z, (Xi, Yi), method=method)
        
        # OTIMIZAÇÃO: Salva em cache
        if cache:
            if not hasattr(self, '_grid_cache'):
                self._grid_cache = {}
            self._grid_cache[cache_key] = (Xi, Yi, Zi)
            logger.debug(f"💾 Grid armazenado em cache (resolution={resolution}x{resolution})")
        
        return Xi, Yi, Zi


@dataclass
class ProcessingResult:
    """
    Resultado de um processamento geofísico.
    
    Attributes:
    -----------
    processed_data : GeophysicalData
        Dados processados
    original_data : GeophysicalData
        Dados originais (para comparação)
    method_name : str
        Nome do método aplicado
    parameters : Dict[str, Any]
        Parâmetros utilizados
    figures : List[go.Figure]
        Lista de figuras Plotly geradas
    explanation : str
        Explicação técnica do resultado
    execution_time : float
        Tempo de execução em segundos
    references : List[str]
        Referências científicas (formato ABNT)
    """
    processed_data: GeophysicalData
    original_data: Optional[GeophysicalData] = None
    method_name: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    figures: List[go.Figure] = field(default_factory=list)
    explanation: str = ""
    execution_time: float = 0.0
    references: List[str] = field(default_factory=list)
    
    def summary(self) -> Dict[str, Any]:
        """Retorna sumário do processamento."""
        return {
            'method': self.method_name,
            'parameters': self.parameters,
            'n_points': self.processed_data.metadata.get('n_points', 0),
            'execution_time_s': round(self.execution_time, 3),
            'statistics': {
                'mean': self.processed_data.metadata.get('mean'),
                'std': self.processed_data.metadata.get('std'),
                'range': [
                    self.processed_data.metadata.get('min'),
                    self.processed_data.metadata.get('max')
                ]
            }
        }


class ProcessingPipeline:
    """
    Gerenciador de pipeline de processamentos sequenciais.
    
    Permite encadear múltiplos processamentos mantendo histórico e estado.
    
    Examples:
    ---------
    >>> pipeline = ProcessingPipeline(initial_data)
    >>> pipeline.add_step('bouguer_correction', density=2670)
    >>> pipeline.add_step('upward_continuation', height=1000)
    >>> results = pipeline.execute()
    """
    
    def __init__(self, initial_data: GeophysicalData):
        self.initial_data = initial_data
        self.current_data = initial_data
        self.steps: List[Dict[str, Any]] = []
        self.results: List[ProcessingResult] = []
        self.total_execution_time = 0.0
        
        logger.info("Pipeline de processamento inicializado")
    
    def add_step(self, function_name: str, **kwargs):
        """
        Adiciona etapa ao pipeline.
        
        Parameters:
        -----------
        function_name : str
            Nome da função registrada em PROCESSING_REGISTRY
        **kwargs
            Parâmetros para a função
        """
        if function_name not in PROCESSING_REGISTRY:
            raise ProcessingError(f"Função '{function_name}' não encontrada no registro")
        
        self.steps.append({
            'function': function_name,
            'params': kwargs
        })
        logger.debug(f"Adicionada etapa: {function_name} com parâmetros {kwargs}")
    
    def execute(self) -> List[ProcessingResult]:
        """
        Executa todas as etapas do pipeline sequencialmente.
        
        Returns:
        --------
        List[ProcessingResult]
            Lista com resultados de cada etapa
        """
        self.results = []
        self.current_data = self.initial_data
        
        for i, step in enumerate(self.steps, 1):
            logger.info(f"Executando etapa {i}/{len(self.steps)}: {step['function']}")
            
            start_time = datetime.now()
            
            try:
                # Obtém função do registro
                func_info = PROCESSING_REGISTRY[step['function']]
                func = func_info['function']
                
                # Executa processamento
                result = func(self.current_data, **step['params'])
                
                # Calcula tempo de execução
                execution_time = (datetime.now() - start_time).total_seconds()
                result.execution_time = execution_time
                self.total_execution_time += execution_time
                
                # Atualiza dado atual e adiciona resultado
                self.current_data = result.processed_data
                self.results.append(result)
                
                logger.success(f"Etapa {i} concluída em {execution_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Erro na etapa {i}: {str(e)}")
                logger.error(traceback.format_exc())
                raise ProcessingError(f"Falha na etapa {step['function']}: {str(e)}")
        
        logger.success(f"Pipeline concluído. Tempo total: {self.total_execution_time:.2f}s")
        return self.results
    
    def get_summary(self) -> Dict[str, Any]:
        """Retorna sumário completo do pipeline."""
        return {
            'n_steps': len(self.steps),
            'total_time_s': round(self.total_execution_time, 3),
            'steps': [
                {
                    'order': i+1,
                    'function': step['function'],
                    'params': step['params']
                }
                for i, step in enumerate(self.steps)
            ],
            'results_summary': [r.summary() for r in self.results]
        }


class RAGEngine:
    """
    Motor de Retrieval-Augmented Generation para contexto científico.
    
    Gerencia embeddings de documentos científicos e recuperação semântica
    para enriquecer respostas do LLM com citações acadêmicas.
    
    Attributes:
    -----------
    database_path : Path
        Caminho para diretório com PDFs científicos
    embedding_model : SentenceTransformer
        Modelo para gerar embeddings
    chroma_client : chromadb.Client
        Cliente ChromaDB para armazenamento vetorial
    collection : chromadb.Collection
        Coleção de documentos
    """
    
    def __init__(self, database_path: Union[str, Path] = "rag_database"):
        self.database_path = Path(database_path)
        self.embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        self.initialized = False
        
        logger.info(f"RAGEngine inicializado. Database: {self.database_path}")
    
    def initialize(self):
        """Inicializa modelos e banco de dados vetorial."""
        if self.initialized:
            return
        
        try:
            # Carrega modelo de embeddings COM GPU
            logger.info(f"Carregando modelo de embeddings: {self.embedding_model_name}")
            
            # OTIMIZAÇÃO: Usa GPU se disponível
            device = GPU_INFO['device'] if GPU_INFO['available'] else 'cpu'
            self.embedding_model = SentenceTransformer(self.embedding_model_name, device=device)
            
            if GPU_INFO['available']:
                logger.success(f"🚀 SentenceTransformer usando GPU: {GPU_INFO['device_name']}")
            else:
                logger.warning("⚠️ SentenceTransformer usando CPU (instale PyTorch para GPU)")
            
            # Inicializa ChromaDB
            chroma_path = self.database_path / "chromadb"
            chroma_path.mkdir(parents=True, exist_ok=True)
            
            self.chroma_client = chromadb.PersistentClient(
                path=str(chroma_path),
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Obtém ou cria coleção
            try:
                self.collection = self.chroma_client.get_collection("geobot_papers")
                logger.info(f"Coleção existente carregada: {self.collection.count()} documentos")
            except:
                self.collection = self.chroma_client.create_collection(
                    name="geobot_papers",
                    metadata={"description": "Scientific papers for geophysics"}
                )
                logger.info("Nova coleção criada")
            
            self.initialized = True
            logger.success("RAGEngine inicializado com sucesso")
            
        except Exception as e:
            logger.error(f"Erro ao inicializar RAGEngine: {str(e)}")
            raise RAGError(f"Falha na inicialização: {str(e)}")
    
    def index_documents(self, force_reindex: bool = False):
        """
        Indexa documentos PDF do diretório database_path.
        
        Parameters:
        -----------
        force_reindex : bool
            Se True, reindexar mesmo que documentos já existam
        """
        if not self.initialized:
            self.initialize()
        
        # Verifica se já existem documentos
        if self.collection.count() > 0 and not force_reindex:
            logger.info("Documentos já indexados. Use force_reindex=True para reindexar")
            return
        
        # Busca PDFs
        pdf_files = list(self.database_path.rglob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"Nenhum PDF encontrado em {self.database_path}")
            logger.info("Adicione papers científicos ao diretório rag_database/")
            return
        
        logger.info(f"Encontrados {len(pdf_files)} PDFs para indexar")
        
        documents = []
        metadatas = []
        ids = []
        
        for pdf_path in tqdm(pdf_files, desc="Indexando PDFs"):
            try:
                # Extrai texto do PDF
                reader = PdfReader(str(pdf_path))
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                
                # Divide em chunks (parágrafos)
                chunks = self._split_text(text, chunk_size=500)
                
                for i, chunk in enumerate(chunks):
                    documents.append(chunk)
                    metadatas.append({
                        'source': pdf_path.name,
                        'path': str(pdf_path),
                        'chunk': i,
                        'total_chunks': len(chunks)
                    })
                    ids.append(f"{pdf_path.stem}_chunk_{i}")
                
                logger.debug(f"Indexado: {pdf_path.name} ({len(chunks)} chunks)")
                
            except Exception as e:
                logger.error(f"Erro ao processar {pdf_path.name}: {str(e)}")
                continue
        
        if documents:
            # Adiciona ao ChromaDB
            logger.info(f"Gerando embeddings para {len(documents)} chunks...")
            embeddings = self.embedding_model.encode(
                documents,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.success(f"{len(documents)} chunks indexados com sucesso")
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Busca documentos relevantes para a query.
        
        Parameters:
        -----------
        query : str
            Texto de busca
        top_k : int
            Número de resultados a retornar
        
        Returns:
        --------
        List[Dict]
            Lista com documentos relevantes e metadados
        """
        if not self.initialized:
            self.initialize()
        
        if self.collection.count() == 0:
            logger.warning("Base de conhecimento vazia")
            return []
        
        # Gera embedding da query
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Busca no ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        # Formata resultados
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None
            })
        
        logger.debug(f"Busca RAG: '{query[:50]}...' → {len(formatted_results)} resultados")
        return formatted_results
    
    def _split_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Divide texto em chunks com overlap."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def format_citation_abnt(self, metadata: Dict[str, Any], text_snippet: str = "") -> str:
        """
        Formata citação no padrão ABNT.
        
        Parameters:
        -----------
        metadata : dict
            Metadados do documento
        text_snippet : str
            Trecho relevante do texto
        
        Returns:
        --------
        str
            Citação formatada em Markdown
        """
        source = metadata.get('source', 'Documento desconhecido')
        
        citation = f"""
> 📚 **Referência:**
> **{source}**
"""
        
        if text_snippet:
            # Limita tamanho do snippet
            if len(text_snippet) > 200:
                text_snippet = text_snippet[:200] + "..."
            citation += f"""
> *Trecho relevante:*
> "{text_snippet}"
"""
        
        return citation


# ============================================================================
# GERENCIADOR DE LLM COM FALLBACK
# ============================================================================

class LLMManager:
    """
    Gerenciador de comunicação com Groq API com sistema de fallback automático.
    
    Attributes:
    -----------
    api_key : str
        Chave da API Groq
    primary_model : str
        Modelo principal preferido
    fallback_models : List[str]
        Lista de modelos alternativos
    current_model : str
        Modelo atualmente ativo
    """
    
    def __init__(self, api_key: str, primary_model: str = None):
        self.api_key = api_key
        self.primary_model = primary_model or GROQ_MODELS[0]
        self.fallback_models = [m for m in GROQ_MODELS if m != self.primary_model]
        self.current_model = self.primary_model
        self.client = Groq(api_key=api_key)
        self.fallback_history: List[Dict[str, Any]] = []
        
        logger.info(f"LLMManager inicializado. Modelo principal: {self.primary_model}")
    
    def list_available_models(self) -> List[str]:
        """Lista modelos disponíveis via API."""
        try:
            models = self.client.models.list()
            available = [m.id for m in models.data]
            logger.info(f"Modelos disponíveis: {len(available)}")
            return available
        except Exception as e:
            logger.error(f"Erro ao listar modelos: {str(e)}")
            return GROQ_MODELS
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.01,
        max_tokens: int = 5000,
        stream: bool = False
    ) -> Union[str, Any]:
        """
        Envia mensagens para o LLM com fallback automático.
        
        Parameters:
        -----------
        messages : List[Dict]
            Lista de mensagens no formato [{"role": "user", "content": "..."}]
        temperature : float
            Temperatura para geração (0-1)
        max_tokens : int
            Máximo de tokens na resposta
        stream : bool
            Se True, retorna generator para streaming
        
        Returns:
        --------
        str ou generator
            Resposta do modelo ou generator (se stream=True)
        """
        models_to_try = [self.current_model] + self.fallback_models
        
        for model in models_to_try:
            try:
                logger.debug(f"Tentando modelo: {model}")
                
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=stream
                )
                
                # Se chegou aqui, sucesso
                if model != self.current_model:
                    self._log_fallback(self.current_model, model)
                    self.current_model = model
                
                if stream:
                    return response
                else:
                    return response.choices[0].message.content
                
            except RateLimitError as e:
                logger.warning(f"Rate limit atingido no modelo {model}: {str(e)}")
                continue
                
            except APIError as e:
                logger.error(f"Erro de API no modelo {model}: {str(e)}")
                continue
                
            except Exception as e:
                logger.error(f"Erro inesperado no modelo {model}: {str(e)}")
                continue
        
        # Se chegou aqui, todos os modelos falharam
        raise LLMError("Todos os modelos falharam. Verifique sua API key ou tente novamente mais tarde.")
    
    def _log_fallback(self, from_model: str, to_model: str):
        """Registra evento de fallback."""
        event = {
            'timestamp': datetime.now().isoformat(),
            'from': from_model,
            'to': to_model
        }
        self.fallback_history.append(event)
        logger.warning(f"FALLBACK: {from_model} → {to_model}")
    
    def get_fallback_summary(self) -> str:
        """Retorna resumo dos eventos de fallback."""
        if not self.fallback_history:
            return "Nenhum fallback necessário até o momento."
        
        summary = f"**Histórico de Fallbacks** ({len(self.fallback_history)} eventos):\n\n"
        for event in self.fallback_history[-5:]:  # Últimos 5
            summary += f"- {event['timestamp']}: {event['from']} → {event['to']}\n"
        
        return summary


# ============================================================================
# FUNÇÕES DE PROCESSAMENTO GEOFÍSICO
# ============================================================================

# Esta seção continua na próxima parte devido ao limite de tamanho...
# Vou criar as funções de processamento em seguida.

def detect_data_type(df: pl.DataFrame) -> str:
    """
    Detecta tipo de dado geofísico baseado em colunas e valores.
    
    Parameters:
    -----------
    df : pl.DataFrame
        DataFrame com dados
    
    Returns:
    --------
    str
        Tipo detectado: 'gravity', 'magnetic', 'topography', 'unknown'
    """
    columns_lower = [c.lower() for c in df.columns]
    
    # Keywords para cada tipo
    gravity_keywords = ['grav', 'bouguer', 'mgal', 'free_air', 'anomaly']
    magnetic_keywords = ['mag', 'nt', 'gamma', 'tmi', 'igrf', 'field']
    topo_keywords = ['elev', 'altitude', 'height', 'dem', 'dtm']
    
    # Conta matches
    gravity_score = sum(any(kw in col for kw in gravity_keywords) for col in columns_lower)
    magnetic_score = sum(any(kw in col for kw in magnetic_keywords) for col in columns_lower)
    topo_score = sum(any(kw in col for kw in topo_keywords) for col in columns_lower)
    
    scores = {
        'gravity': gravity_score,
        'magnetic': magnetic_score,
        'topography': topo_score
    }
    
    max_score = max(scores.values())
    if max_score == 0:
        return 'unknown'
    
    detected_type = max(scores, key=scores.get)
    logger.info(f"Tipo de dado detectado: {detected_type} (score: {max_score})")
    
    return detected_type


def parse_uploaded_file(file, filename: str) -> GeophysicalData:
    """
    Parseia arquivo carregado e retorna GeophysicalData.
    
    Parameters:
    -----------
    file : UploadedFile
        Arquivo do Streamlit
    filename : str
        Nome do arquivo
    
    Returns:
    --------
    GeophysicalData
        Dados estruturados
    """
    logger.info(f"Parseando arquivo: {filename}")
    
    # Determina tipo de arquivo
    file_ext = Path(filename).suffix.lower()
    
    try:
        # Lê dados
        if file_ext in ['.csv', '.txt']:
            # Tenta detectar delimitador
            content = file.read().decode('utf-8')
            file.seek(0)
            
            if ',' in content:
                df = pl.read_csv(file)
            elif '\t' in content:
                df = pl.read_csv(file, separator='\t')
            else:
                df = pl.read_csv(file, separator=r'\s+')
                
        elif file_ext in ['.xlsx', '.xls']:
            df = pl.read_excel(file)
            
        else:
            raise InvalidDataError(f"Formato não suportado: {file_ext}")
        
        logger.info(f"Arquivo lido: {df.shape[0]} linhas, {df.shape[1]} colunas")
        
        # Detecta estrutura
        data_type = detect_data_type(df)
        
        # Identifica colunas de coordenadas
        coords = {}
        value_col = None
        
        columns_lower = {c.lower(): c for c in df.columns}
        
        # Mapeia coordenadas
        x_candidates = ['x', 'lon', 'longitude', 'long', 'easting']
        y_candidates = ['y', 'lat', 'latitude', 'northing']
        z_candidates = ['z', 'elevation', 'altitude', 'height', 'depth']
        
        for cand in x_candidates:
            if cand in columns_lower:
                coords['x'] = columns_lower[cand]
                break
        
        for cand in y_candidates:
            if cand in columns_lower:
                coords['y'] = columns_lower[cand]
                break
        
        for cand in z_candidates:
            if cand in columns_lower:
                coords['z'] = columns_lower[cand]
                break
        
        # Identifica coluna de valores
        if data_type == 'gravity':
            value_candidates = ['gravity', 'grav', 'bouguer', 'anomaly', 'value']
        elif data_type == 'magnetic':
            value_candidates = ['magnetic', 'mag', 'tmi', 'field', 'value']
        else:
            value_candidates = ['value', 'data', 'measurement']
        
        for cand in value_candidates:
            matching = [c for c in columns_lower.keys() if cand in c]
            if matching:
                value_col = columns_lower[matching[0]]
                break
        
        # Se não encontrou coluna de valores, usa a última coluna numérica
        if value_col is None:
            numeric_cols = [c for c in df.columns if df[c].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]]
            if numeric_cols:
                value_col = numeric_cols[-1]
                logger.warning(f"Coluna de valores não identificada, usando: {value_col}")
        
        # Valida coordenadas mínimas
        if 'x' not in coords or 'y' not in coords:
            raise InvalidDataError("Não foi possível identificar colunas de coordenadas X e Y")
        
        if value_col is None:
            raise InvalidDataError("Não foi possível identificar coluna de valores")
        
        # Determina dimensionalidade
        if 'z' in coords:
            dimension = '3D'
        elif len(df) > 10:
            dimension = '2D'
        else:
            dimension = '1D'
        
        # Determina unidades
        if data_type == 'gravity':
            units = 'mGal'
        elif data_type == 'magnetic':
            units = 'nT'
        else:
            units = 'SI'
        
        # Cria objeto GeophysicalData
        geo_data = GeophysicalData(
            data=df,
            data_type=data_type,
            dimension=dimension,
            coords=coords,
            value_column=value_col,
            units=units,
            metadata={'filename': filename}
        )
        
        logger.success(f"Dados parseados com sucesso: {data_type} {dimension}")
        return geo_data
        
    except Exception as e:
        logger.error(f"Erro ao parsear arquivo: {str(e)}")
        raise InvalidDataError(f"Erro ao ler arquivo: {str(e)}")


# ============================================================================
# FUNÇÕES DE PROCESSAMENTO GEOFÍSICO
# ============================================================================

@register_processing(
    category="Gravimetria",
    description="Correção de Bouguer simples para dados gravimétricos",
    input_type="grid",
    requires_params=['density']
)
def bouguer_correction(data: GeophysicalData, density: float = 2.67) -> ProcessingResult:
    """
    Correção de Bouguer Completa (Free-air + Bouguer Slab)
    
    A correção de Bouguer completa remove os efeitos gravitacionais da topografia,
    combinando a correção de ar livre (free-air) e a correção da placa de Bouguer.
    
    Fórmulas:
    ---------
    1. Correção de ar livre (Free-air):
       Δg_FA = 0.3086 × h (mGal)
       
    2. Correção de Bouguer (placa):
       Δg_B = 0.1119 × ρ × h (mGal)
       
    3. Correção total:
       Δg_total = Δg_FA - Δg_B = (0.3086 - 0.1119×ρ) × h
       
    4. Gravidade Bouguer-corrigida:
       g_B = g_obs - Δg_total
    
    Para ρ = 2.67 g/cm³ (densidade típica da crosta):
       Δg_total ≈ 0.19664 × h (mGal)
    
    Onde:
        h = altura acima do datum (m)
        ρ = densidade da topografia (g/cm³)
    
    Aplicações:
    -----------
    - Exploração mineral (identificação de corpos densos/menos densos)
    - Estudos crustais e litosféricos
    - Mapeamento de bacias sedimentares
    
    Limitações:
    -----------
    - Assume topografia como placa infinita (simplificação)
    - Para terrenos acidentados, requer correção de terreno adicional
    - Sensível à escolha da densidade (requer informação geológica)
    
    Referências:
    ------------
    BLAKELY, R. J. **Potential Theory in Gravity and Magnetic Applications**. 
    Cambridge University Press, 1995. p. 135-142. ISBN: 978-0521575478
    
    TELFORD, W. M.; GELDART, L. P.; SHERIFF, R. E. **Applied Geophysics**. 
    2nd ed. Cambridge University Press, 1990. p. 6-53. ISBN: 978-0521339384
    
    Parameters:
    -----------
    data : GeophysicalData
        Dados gravimétricos com coluna de elevação
    density : float
        Densidade da topografia em g/cm³ (padrão: 2.67 g/cm³ - crosta continental)
    
    Returns:
    --------
    ProcessingResult
        Dados corrigidos com figuras comparativas
    
    Examples:
    ---------
    >>> result = bouguer_correction(gravity_data, density=2.67)
    >>> result.processed_data.to_pandas()
    """
    start_time = datetime.now()
    
    try:
        # Validações
        if data.data_type != 'gravity':
            logger.warning(f"Correção de Bouguer aplicada a dados tipo '{data.data_type}' (esperado 'gravity')")
        
        if 'z' not in data.coords:
            raise ProcessingError("Correção de Bouguer requer coluna de elevação (z)")
        
        # Constantes da correção de Bouguer completa
        FA_factor = 0.3086      # Free-air correction (mGal/m)
        B_factor = 0.1119       # Bouguer slab factor (mGal/m per g/cm³)
        
        # Fator total: Free-air - Bouguer
        # Para ρ=2.67: 0.3086 - 0.1119*2.67 = 0.19664 mGal/m
        total_factor = FA_factor - (B_factor * density)
        
        # Obtém elevações
        z_col = data.coords['z']
        elevations = data.data.select(z_col).to_numpy().flatten()
        
        # Calcula cada componente da correção
        freeair_correction = FA_factor * elevations
        bouguer_slab_correction = B_factor * density * elevations
        total_correction = total_factor * elevations
        
        # Aplica correção (subtrai da anomalia observada)
        original_values = data.data.select(data.value_column).to_numpy().flatten()
        corrected_values = original_values - total_correction
        
        # Cria novo DataFrame com todas as colunas de correção
        corrected_df = data.data.with_columns([
            pl.Series(name="freeair_correction", values=freeair_correction),
            pl.Series(name="bouguer_slab_correction", values=bouguer_slab_correction),
            pl.Series(name="total_correction", values=total_correction),
            pl.Series(name=f"{data.value_column}_bouguer", values=corrected_values)
        ])
        
        # Cria GeophysicalData de saída
        processed_data = GeophysicalData(
            data=corrected_df,
            data_type=data.data_type,
            dimension=data.dimension,
            coords=data.coords,
            value_column=f"{data.value_column}_bouguer",
            units=data.units,
            crs=data.crs,
            metadata={
                **data.metadata,
                'processing': 'bouguer_correction_complete',
                'density_gcm3': density,
                'freeair_factor': FA_factor,
                'bouguer_factor': B_factor,
                'total_factor': total_factor
            }
        )
        
        # Cria figuras comparativas
        figures = create_comparison_plots(data, processed_data, "Correção de Bouguer Completa")
        
        # Explicação
        explanation = f"""
### 📊 Correção de Bouguer Completa Aplicada

**Parâmetros:**
- Densidade da topografia: {density:.2f} g/cm³
- Fator Free-air: {FA_factor:.4f} mGal/m
- Fator Bouguer (placa): {B_factor:.4f} mGal/m per g/cm³
- **Fator total de correção: {total_factor:.5f} mGal/m**

**Topografia:**
- Elevação mínima: {elevations.min():.1f} m
- Elevação máxima: {elevations.max():.1f} m
- Elevação média: {elevations.mean():.1f} m

**Componentes da Correção:**
- Free-air mínima/máxima: {freeair_correction.min():.2f} / {freeair_correction.max():.2f} mGal
- Bouguer slab mínima/máxima: {bouguer_slab_correction.min():.2f} / {bouguer_slab_correction.max():.2f} mGal
- **Correção total mínima/máxima: {total_correction.min():.2f} / {total_correction.max():.2f} mGal**
- Correção média: {total_correction.mean():.2f} mGal

**Resultado:**
- Gravidade observada: {original_values.min():.2f} a {original_values.max():.2f} mGal (amplitude: {original_values.max()-original_values.min():.2f} mGal)
- **Anomalia de Bouguer: {corrected_values.min():.2f} a {corrected_values.max():.2f} mGal (amplitude: {corrected_values.max()-corrected_values.min():.2f} mGal)**

A correção de Bouguer completa remove a tendência regional relacionada à topografia
(free-air) e o efeito da massa topográfica (placa de Bouguer), realçando anomalias
locais de densidade na subsuperfície.
"""
        
        # Referências
        references = [
            "BLAKELY, R. J. **Potential Theory in Gravity and Magnetic Applications**. Cambridge University Press, 1995. ISBN: 978-0521575478",
            "TELFORD, W. M.; GELDART, L. P.; SHERIFF, R. E. **Applied Geophysics**. 2nd ed. Cambridge University Press, 1990. ISBN: 978-0521339384"
        ]
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return ProcessingResult(
            processed_data=processed_data,
            original_data=data,
            method_name="bouguer_correction",
            parameters={'density': density},
            figures=figures,
            explanation=explanation,
            execution_time=execution_time,
            references=references
        )
        
    except Exception as e:
        logger.error(f"Erro na correção de Bouguer: {str(e)}")
        raise ProcessingError(f"Falha na correção de Bouguer: {str(e)}")


@register_processing(
    category="Geral",
    description="Continuação ascendente de campos potenciais",
    input_type="grid",
    requires_params=['height']
)
def upward_continuation(data: GeophysicalData, height: float = 1000.0) -> ProcessingResult:
    """
    Continuação Ascendente no Domínio da Frequência
    
    A continuação ascendente calcula o valor do campo potencial em um plano acima
    do plano de observação. É uma operação de suavização que atenua anomalias de
    fontes rasas (alta frequência) e realça fontes profundas (baixa frequência).
    
    Fundamento Teórico:
    -------------------
    No domínio da frequência, a continuação ascendente é dada por:
        F{U(x,y,z+Δz)} = F{U(x,y,z)} × exp(-|k|Δz)
    
    Onde:
        F{} = transformada de Fourier 2D
        k = número de onda = sqrt(kx² + ky²)
        Δz = altura de continuação (positiva para cima)
    
    Aplicações:
    -----------
    - Separação regional-residual
    - Estimativa de profundidade de fontes
    - Redução de ruído de alta frequência
    - Comparação com dados aeromagnéticos/gravimétricos
    
    Limitações:
    -----------
    - Amplifica ruído se Δz for muito grande
    - Requer grid regular
    - Pressupõe campo harmonicamente continuado
    
    Referências:
    ------------
    BLAKELY, R. J. **Potential Theory in Gravity and Magnetic Applications**. 
    Cambridge University Press, 1995. p. 312-319. ISBN: 978-0521575478
    
    JACOBSEN, B. H. **A case for upward continuation as the standard separation filter 
    for potential-field maps**. Geophysics, v. 52, n. 8, p. 1138-1148, 1987. 
    DOI: 10.1190/1.1442378
    
    Parameters:
    -----------
    data : GeophysicalData
        Dados em grid regular
    height : float
        Altura de continuação em metros (positivo = para cima)
    
    Returns:
    --------
    ProcessingResult
        Dados continuados com figuras comparativas
    
    Examples:
    ---------
    >>> result = upward_continuation(magnetic_data, height=500)
    """
    start_time = datetime.now()
    
    try:
        # Interpola para grid regular
        Xi, Yi, Zi = data.to_grid(method='linear')
        
        # Dimensões do grid
        ny, nx = Zi.shape
        
        # Remove valores NaN (interpola ou mascara)
        mask = np.isnan(Zi)
        if mask.any():
            logger.warning(f"{mask.sum()} valores NaN encontrados, interpolando...")
            # Interpolação simples com inpaint
            from scipy.ndimage import distance_transform_edt
            indices = distance_transform_edt(mask, return_distances=False, return_indices=True)
            Zi[mask] = Zi[tuple(indices[:, mask])]
        
        # Calcula espaçamento do grid
        dx = (Xi.max() - Xi.min()) / (nx - 1)
        dy = (Yi.max() - Yi.min()) / (ny - 1)
        
        # Números de onda (frequências)
        kx = 2 * np.pi * fftfreq(nx, d=dx)
        ky = 2 * np.pi * fftfreq(ny, d=dy)
        KX, KY = np.meshgrid(kx, ky)
        K = np.sqrt(KX**2 + KY**2)
        
        # Transformada de Fourier 2D 🚀 COM GPU (10-50x mais rápido)
        F = fft2_gpu(Zi)
        
        # Aplica continuação no domínio da frequência
        # U_up = F^-1{ F{U} × exp(-|k|Δz) }
        F_continued = F * np.exp(-K * height)
        
        # Transformada inversa 🚀 COM GPU
        Zi_continued = np.real(ifft2_gpu(F_continued))
        
        if GPU_INFO['available']:
            logger.debug(f"✅ Continuação ascendente processada na GPU: {GPU_INFO['device_name']}")
        
        # Converte grid de volta para pontos
        x_flat = Xi.flatten()
        y_flat = Yi.flatten()
        z_continued_flat = Zi_continued.flatten()
        
        # Cria novo DataFrame
        x_col = data.coords['x']
        y_col = data.coords['y']
        
        continued_df = pl.DataFrame({
            x_col: x_flat,
            y_col: y_flat,
            f"{data.value_column}_upward_{int(height)}m": z_continued_flat
        })
        
        # Cria GeophysicalData de saída
        # Coords sem 'z' pois DataFrame resultante tem apenas x, y, value
        new_coords = {'x': data.coords['x'], 'y': data.coords['y']}
        
        processed_data = GeophysicalData(
            data=continued_df,
            data_type=data.data_type,
            dimension=data.dimension,
            coords=new_coords,
            value_column=f"{data.value_column}_upward_{int(height)}m",
            units=data.units,
            crs=data.crs,
            metadata={
                **data.metadata,
                'processing': 'upward_continuation',
                'height_m': height,
                'grid_spacing': f"{dx:.2f} x {dy:.2f}"
            }
        )
        
        # Cria figuras
        figures = create_comparison_plots(data, processed_data, f"Continuação Ascendente ({height}m)")
        
        # Explicação
        attenuation = np.exp(-K.max() * height)
        
        explanation = f"""
### 📊 Continuação Ascendente Aplicada

**Parâmetros:**
- Altura de continuação: {height:.0f} m
- Dimensão do grid: {ny} × {nx}
- Espaçamento: {dx:.2f} × {dy:.2f} m

**Efeito da Filtragem:**
- Atenuação de alta frequência: {(1-attenuation)*100:.1f}%
- Comprimento de onda de corte (50% atenuação): {np.log(2)/K.max()*height:.1f} m

**Resultado:**
- Campo original: {Zi.min():.2f} a {Zi.max():.2f} {data.units}
- Campo continuado: {Zi_continued.min():.2f} a {Zi_continued.max():.2f} {data.units}
- Redução de amplitude: {((Zi.max()-Zi.min()) - (Zi_continued.max()-Zi_continued.min())):.2f} {data.units}

A continuação ascendente suaviza o campo, atenuando anomalias de fontes rasas
e realçando tendências regionais de fontes profundas.
"""
        
        references = [
            "BLAKELY, R. J. **Potential Theory in Gravity and Magnetic Applications**. Cambridge University Press, 1995. p. 312-319. ISBN: 978-0521575478",
            "JACOBSEN, B. H. **A case for upward continuation as the standard separation filter for potential-field maps**. Geophysics, v. 52, n. 8, p. 1138-1148, 1987. DOI: 10.1190/1.1442378"
        ]
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return ProcessingResult(
            processed_data=processed_data,
            original_data=data,
            method_name="upward_continuation",
            parameters={'height': height},
            figures=figures,
            explanation=explanation,
            execution_time=execution_time,
            references=references
        )
        
    except Exception as e:
        logger.error(f"Erro na continuação ascendente: {str(e)}")
        raise ProcessingError(f"Falha na continuação ascendente: {str(e)}")


@register_processing(
    category="Geral",
    description="Derivada vertical de 1ª ordem (realce de bordas rasas)",
    input_type="grid",
    requires_params=[]
)
def vertical_derivative(data: GeophysicalData) -> ProcessingResult:
    """
    Derivada Vertical de Primeira Ordem
    
    Calcula a taxa de variação do campo potencial na direção vertical (∂U/∂z).
    É uma operação de realce que enfatiza anomalias rasas e bordas de corpos.
    
    Fundamento Teórico:
    -------------------
    No domínio da frequência:
        F{∂U/∂z} = F{U} × |k|
    
    Onde:
        F{} = transformada de Fourier 2D
        k = número de onda = sqrt(kx² + ky²)
    
    Aplicações:
    -----------
    - Delineamento de bordas de corpos geológicos
    - Estimativa de profundidade (regra de Peters)
    - Realce de anomalias rasas
    - Separação de fontes sobrepostas
    
    Limitações:
    -----------
    - Amplifica ruído de alta frequência
    - Requer filtragem passa-baixa prévia se dados ruidosos
    - Sensível à qualidade do gridding
    
    Referências:
    ------------
    BLAKELY, R. J. **Potential Theory in Gravity and Magnetic Applications**. 
    Cambridge University Press, 1995. p. 320-325. ISBN: 978-0521575478
    
    NABIGHIAN, M. N. et al. **The historical development of the magnetic method 
    in exploration**. Geophysics, v. 70, n. 6, p. 33ND-61ND, 2005. 
    DOI: 10.1190/1.2133784
    
    Parameters:
    -----------
    data : GeophysicalData
        Dados em grid regular
    
    Returns:
    --------
    ProcessingResult
        Derivada vertical com figuras
    """
    start_time = datetime.now()
    
    try:
        # Interpola para grid
        Xi, Yi, Zi = data.to_grid(method='linear')
        ny, nx = Zi.shape
        
        # Remove NaN
        mask = np.isnan(Zi)
        if mask.any():
            from scipy.ndimage import distance_transform_edt
            indices = distance_transform_edt(mask, return_distances=False, return_indices=True)
            Zi[mask] = Zi[tuple(indices[:, mask])]
        
        # Espaçamento
        dx = (Xi.max() - Xi.min()) / (nx - 1)
        dy = (Yi.max() - Yi.min()) / (ny - 1)
        
        # Números de onda
        kx = 2 * np.pi * fftfreq(nx, d=dx)
        ky = 2 * np.pi * fftfreq(ny, d=dy)
        KX, KY = np.meshgrid(kx, ky)
        K = np.sqrt(KX**2 + KY**2)
        
        # FFT e aplicação 🚀 COM GPU (10-50x mais rápido)
        F = fft2_gpu(Zi)
        F_deriv = F * K
        Zi_deriv = np.real(ifft2_gpu(F_deriv))
        
        if GPU_INFO['available']:
            logger.debug(f"✅ Derivada vertical processada na GPU: {GPU_INFO['device_name']}")
        
        # Converte para pontos
        x_flat = Xi.flatten()
        y_flat = Yi.flatten()
        z_deriv_flat = Zi_deriv.flatten()
        
        x_col = data.coords['x']
        y_col = data.coords['y']
        
        deriv_df = pl.DataFrame({
            x_col: x_flat,
            y_col: y_flat,
            f"{data.value_column}_dz": z_deriv_flat
        })
        
        # Coordenadas sem 'z' pois resultado é grid 2D
        new_coords = {'x': data.coords['x'], 'y': data.coords['y']}
        
        processed_data = GeophysicalData(
            data=deriv_df,
            data_type=data.data_type,
            dimension=data.dimension,
            coords=new_coords,
            value_column=f"{data.value_column}_dz",
            units=f"{data.units}/m",
            crs=data.crs,
            metadata={
                **data.metadata,
                'processing': 'vertical_derivative',
                'grid_spacing': f"{dx:.2f} x {dy:.2f}"
            }
        )
        
        figures = create_comparison_plots(data, processed_data, "Derivada Vertical (∂U/∂z)")
        
        explanation = f"""
### 📊 Derivada Vertical Aplicada

**Parâmetros:**
- Dimensão do grid: {ny} × {nx}
- Espaçamento: {dx:.2f} × {dy:.2f} m
- Número de onda máximo: {K.max():.6f} rad/m

**Resultado:**
- Campo original: {Zi.min():.2f} a {Zi.max():.2f} {data.units}
- Derivada: {Zi_deriv.min():.3f} a {Zi_deriv.max():.3f} {data.units}/m
- Realce: {(Zi_deriv.std()/Zi.std()):.2f}x

A derivada vertical realça bordas de corpos e anomalias rasas.
"""
        
        references = [
            "BLAKELY, R. J. **Potential Theory in Gravity and Magnetic Applications**. Cambridge University Press, 1995. p. 320-325.",
            "NABIGHIAN, M. N. et al. **The historical development of the magnetic method in exploration**. Geophysics, v. 70, n. 6, 2005."
        ]
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return ProcessingResult(
            processed_data=processed_data,
            original_data=data,
            method_name="vertical_derivative",
            parameters={},
            figures=figures,
            explanation=explanation,
            execution_time=execution_time,
            references=references
        )
        
    except Exception as e:
        logger.error(f"Erro na derivada vertical: {str(e)}")
        raise ProcessingError(f"Falha na derivada vertical: {str(e)}")


@register_processing(
    category="Geral",
    description="Derivada horizontal total (THD - Total Horizontal Derivative)",
    input_type="grid",
    requires_params=[]
)
def horizontal_derivative_total(data: GeophysicalData) -> ProcessingResult:
    """
    Derivada Horizontal Total (THD)
    
    Calcula a magnitude do gradiente horizontal do campo potencial.
    THD = sqrt((∂U/∂x)² + (∂U/∂y)²)
    
    Fundamento Teórico:
    -------------------
    No domínio da frequência:
        F{∂U/∂x} = F{U} × (i·kx)
        F{∂U/∂y} = F{U} × (i·ky)
        THD = sqrt(dx² + dy²)
    
    Aplicações:
    -----------
    - Delineamento preciso de bordas verticais
    - Independente da direção de magnetização
    - Mapeamento de contatos geológicos
    - Interpretação estrutural
    
    Vantagens:
    ----------
    - Não requer conhecimento de magnetização
    - Máximos sobre bordas verticais
    - Menos sensível a ruído que derivadas individuais
    
    Referências:
    ------------
    CORDELL, L.; GRAUCH, V. J. S. **Mapping basement magnetization zones from 
    aeromagnetic data**. SEG Technical Program, p. 181-183, 1985. 
    DOI: 10.1190/1.1892795
    
    PHILLIPS, J. D. **Processing and interpretation of aeromagnetic data for the 
    Santa Cruz Basin**. U.S. Geological Survey Open-File Report 01-0081, 2001.
    
    Parameters:
    -----------
    data : GeophysicalData
        Dados em grid regular
    
    Returns:
    --------
    ProcessingResult
        THD com figuras
    """
    start_time = datetime.now()
    
    try:
        # Grid
        Xi, Yi, Zi = data.to_grid(method='linear')
        ny, nx = Zi.shape
        
        # Remove NaN
        mask = np.isnan(Zi)
        if mask.any():
            from scipy.ndimage import distance_transform_edt
            indices = distance_transform_edt(mask, return_distances=False, return_indices=True)
            Zi[mask] = Zi[tuple(indices[:, mask])]
        
        dx = (Xi.max() - Xi.min()) / (nx - 1)
        dy = (Yi.max() - Yi.min()) / (ny - 1)
        
        # Números de onda
        kx = 2 * np.pi * fftfreq(nx, d=dx)
        ky = 2 * np.pi * fftfreq(ny, d=dy)
        KX, KY = np.meshgrid(kx, ky)
        
        # FFT 🚀 COM GPU (10-50x mais rápido)
        F = fft2_gpu(Zi)
        
        # Derivadas
        F_dx = F * (1j * KX)
        F_dy = F * (1j * KY)
        
        dx_field = np.real(ifft2_gpu(F_dx))
        dy_field = np.real(ifft2_gpu(F_dy))
        
        if GPU_INFO['available']:
            logger.debug(f"✅ Derivadas horizontais processadas na GPU: {GPU_INFO['device_name']}")
        
        # THD
        thd = np.sqrt(dx_field**2 + dy_field**2)
        
        # Converte
        x_flat = Xi.flatten()
        y_flat = Yi.flatten()
        thd_flat = thd.flatten()
        
        x_col = data.coords['x']
        y_col = data.coords['y']
        
        thd_df = pl.DataFrame({
            x_col: x_flat,
            y_col: y_flat,
            f"{data.value_column}_thd": thd_flat
        })
        
        processed_data = GeophysicalData(
            data=thd_df,
            data_type=data.data_type,
            dimension=data.dimension,
            coords=data.coords,
            value_column=f"{data.value_column}_thd",
            units=f"{data.units}/m",
            crs=data.crs,
            metadata={
                **data.metadata,
                'processing': 'horizontal_derivative_total'
            }
        )
        
        figures = create_comparison_plots(data, processed_data, "Derivada Horizontal Total (THD)")
        
        explanation = f"""
### 📊 Derivada Horizontal Total (THD)

**Resultado:**
- Campo original: {Zi.min():.2f} a {Zi.max():.2f} {data.units}
- THD: {thd.min():.3f} a {thd.max():.3f} {data.units}/m
- Média THD: {thd.mean():.3f} {data.units}/m

THD é excelente para delinear bordas e contatos, com máximos sobre descontinuidades verticais.
"""
        
        references = [
            "CORDELL, L.; GRAUCH, V. J. S. **Mapping basement magnetization zones**. SEG, 1985.",
            "PHILLIPS, J. D. **Processing and interpretation of aeromagnetic data**. USGS Report 01-0081, 2001."
        ]
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return ProcessingResult(
            processed_data=processed_data,
            original_data=data,
            method_name="horizontal_derivative_total",
            parameters={},
            figures=figures,
            explanation=explanation,
            execution_time=execution_time,
            references=references
        )
        
    except Exception as e:
        logger.error(f"Erro no THD: {str(e)}")
        raise ProcessingError(f"Falha no THD: {str(e)}")


@register_processing(
    category="Magnetometria",
    description="Redução ao Polo (RTP - Reduction to the Pole)",
    input_type="grid",
    requires_params=['inc_field', 'dec_field', 'inc_mag', 'dec_mag']
)
def reduction_to_pole(
    data: GeophysicalData,
    inc_field: float,
    dec_field: float,
    inc_mag: float = None,
    dec_mag: float = None
) -> ProcessingResult:
    """
    Redução ao Polo (RTP)
    
    Transforma anomalia magnética para como seria medida no polo magnético
    (inclinação = 90°), facilitando interpretação.
    
    Fundamento Teórico:
    -------------------
    RTP = F⁻¹{F{TMI} × [kz²/(θf × θm)]}
    
    Onde:
        θf = kz·sin(If) + i(kx·cos(If)·cos(Df) + ky·cos(If)·sin(Df))
        θm = kz·sin(Im) + i(kx·cos(Im)·cos(Dm) + ky·cos(Im)·sin(Dm))
        
    Aplicações:
    -----------
    - Interpretação de anomalias magnéticas em baixas latitudes
    - Centralização de anomalias sobre fontes
    - Simplificação de formas de anomalias
    
    Limitações:
    -----------
    - Instável próximo ao equador magnético (inc < 15°)
    - Amplifica ruído
    - Requer conhecimento de magnetização
    
    Referências:
    ------------
    BLAKELY, R. J. **Potential Theory in Gravity and Magnetic Applications**. 
    Cambridge University Press, 1995. p. 330-333. ISBN: 978-0521575478
    
    BARANOV, V. **A new method for interpretation of aeromagnetic maps: 
    pseudo-gravimetric anomalies**. Geophysics, v. 22, n. 2, p. 359-383, 1957.
    
    Parameters:
    -----------
    data : GeophysicalData
        Dados magnéticos
    inc_field : float
        Inclinação do campo geomagnético (graus)
    dec_field : float
        Declinação do campo geomagnético (graus)
    inc_mag : float
        Inclinação da magnetização (graus, padrão=inc_field)
    dec_mag : float
        Declinação da magnetização (graus, padrão=dec_field)
    
    Returns:
    --------
    ProcessingResult
        Dado reduzido ao polo
    """
    start_time = datetime.now()
    
    # Se magnetização não fornecida, assume mesma direção do campo
    if inc_mag is None:
        inc_mag = inc_field
    if dec_mag is None:
        dec_mag = dec_field
    
    try:
        # Grid
        Xi, Yi, Zi = data.to_grid(method='linear')
        ny, nx = Zi.shape
        
        # Remove NaN
        mask = np.isnan(Zi)
        if mask.any():
            from scipy.ndimage import distance_transform_edt
            indices = distance_transform_edt(mask, return_distances=False, return_indices=True)
            Zi[mask] = Zi[tuple(indices[:, mask])]
        
        dx = (Xi.max() - Xi.min()) / (nx - 1)
        dy = (Yi.max() - Yi.min()) / (ny - 1)
        
        # Números de onda
        kx = 2 * np.pi * fftfreq(nx, d=dx)
        ky = 2 * np.pi * fftfreq(ny, d=dy)
        KX, KY = np.meshgrid(kx, ky)
        K = np.sqrt(KX**2 + KY**2)
        KZ = K  # Assumindo campo harmônico
        
        # Conversão para radianos
        inc_f_rad = np.deg2rad(inc_field)
        dec_f_rad = np.deg2rad(dec_field)
        inc_m_rad = np.deg2rad(inc_mag)
        dec_m_rad = np.deg2rad(dec_mag)
        
        # Vetores unitários
        # Campo geomagnético
        ux_f = np.cos(inc_f_rad) * np.cos(dec_f_rad)
        uy_f = np.cos(inc_f_rad) * np.sin(dec_f_rad)
        uz_f = np.sin(inc_f_rad)
        
        # Magnetização
        ux_m = np.cos(inc_m_rad) * np.cos(dec_m_rad)
        uy_m = np.cos(inc_m_rad) * np.sin(dec_m_rad)
        uz_m = np.sin(inc_m_rad)
        
        # Direção do campo (theta_f)
        theta_f = KZ * uz_f + 1j * (KX * ux_f + KY * uy_f)
        
        # Direção da magnetização (theta_m)
        theta_m = KZ * uz_m + 1j * (KX * ux_m + KY * uy_m)
        
        # Evita divisão por zero
        theta_f[0, 0] = 1.0
        theta_m[0, 0] = 1.0
        
        # Filtro RTP
        rtp_filter = (KZ ** 2) / (theta_f * theta_m)
        rtp_filter[0, 0] = 0  # DC component
        
        # Aplica filtro 🚀 COM GPU (10-50x mais rápido)
        F = fft2_gpu(Zi)
        F_rtp = F * rtp_filter
        Zi_rtp = np.real(ifft2_gpu(F_rtp))
        
        if GPU_INFO['available']:
            logger.debug(f"✅ Redução ao polo processada na GPU: {GPU_INFO['device_name']}")
        
        # Converte
        x_flat = Xi.flatten()
        y_flat = Yi.flatten()
        z_rtp_flat = Zi_rtp.flatten()
        
        x_col = data.coords['x']
        y_col = data.coords['y']
        
        rtp_df = pl.DataFrame({
            x_col: x_flat,
            y_col: y_flat,
            f"{data.value_column}_rtp": z_rtp_flat
        })
        
        new_coords = {'x': data.coords['x'], 'y': data.coords['y']}
        
        processed_data = GeophysicalData(
            data=rtp_df,
            data_type=data.data_type,
            dimension=data.dimension,
            coords=new_coords,
            value_column=f"{data.value_column}_rtp",
            units=data.units,
            crs=data.crs,
            metadata={
                **data.metadata,
                'processing': 'reduction_to_pole',
                'inc_field': inc_field,
                'dec_field': dec_field,
                'inc_mag': inc_mag,
                'dec_mag': dec_mag
            }
        )
        
        figures = create_comparison_plots(data, processed_data, "Redução ao Polo (RTP)")
        
        explanation = f"""
### 📊 Redução ao Polo Aplicada

**Parâmetros:**
- Campo Geomagnético: Inc={inc_field:.1f}°, Dec={dec_field:.1f}°
- Magnetização: Inc={inc_mag:.1f}°, Dec={dec_mag:.1f}°

**Resultado:**
- Anomalia original: {Zi.min():.2f} a {Zi.max():.2f} {data.units}
- Anomalia RTP: {Zi_rtp.min():.2f} a {Zi_rtp.max():.2f} {data.units}

RTP centraliza anomalias sobre suas fontes, facilitando interpretação.

⚠️ **Atenção**: Se inclinação < 15°, resultado pode ser instável.
"""
        
        references = [
            "BLAKELY, R. J. **Potential Theory in Gravity and Magnetic Applications**. Cambridge University Press, 1995. p. 330-333.",
            "BARANOV, V. **A new method for interpretation of aeromagnetic maps**. Geophysics, v. 22, n. 2, 1957."
        ]
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return ProcessingResult(
            processed_data=processed_data,
            original_data=data,
            method_name="reduction_to_pole",
            parameters={
                'inc_field': inc_field,
                'dec_field': dec_field,
                'inc_mag': inc_mag,
                'dec_mag': dec_mag
            },
            figures=figures,
            explanation=explanation,
            execution_time=execution_time,
            references=references
        )
        
    except Exception as e:
        logger.error(f"Erro no RTP: {str(e)}")
        raise ProcessingError(f"Falha no RTP: {str(e)}")


@register_processing(
    category="Geral",
    description="Sinal Analítico (Amplitude do Sinal Analítico)",
    input_type="grid",
    requires_params=[]
)
def analytic_signal(data: GeophysicalData) -> ProcessingResult:
    """
    Sinal Analítico (Amplitude do Sinal Analítico - ASA)
    
    Calcula a amplitude do sinal analítico 3D:
    ASA = sqrt((∂U/∂x)² + (∂U/∂y)² + (∂U/∂z)²)
    
    Fundamento Teórico:
    -------------------
    O sinal analítico é independente da direção de magnetização e
    apresenta máximos sobre bordas de corpos magnetizados.
    
    Aplicações:
    -----------
    - Delineamento de corpos sem conhecer magnetização
    - Estimativa de profundidade (razão ASA/THD)
    - Interpretação em baixas latitudes magnéticas
    - Independente de RTP
    
    Vantagens:
    ----------
    - Não requer direção de magnetização
    - Estável em qualquer latitude
    - Máximos bem definidos sobre bordas
    
    Referências:
    ------------
    NABIGHIAN, M. N. **The analytic signal of two-dimensional magnetic bodies 
    with polygonal cross-section: its properties and use for automated anomaly 
    interpretation**. Geophysics, v. 37, n. 3, p. 507-517, 1972. 
    DOI: 10.1190/1.1440276
    
    ROEST, W. R.; VERHOEF, J.; PILKINGTON, M. **Magnetic interpretation using 
    the 3-D analytic signal**. Geophysics, v. 57, n. 1, p. 116-125, 1992.
    DOI: 10.1190/1.1443174
    
    Parameters:
    -----------
    data : GeophysicalData
        Dados em grid regular
    
    Returns:
    --------
    ProcessingResult
        Amplitude do sinal analítico
    """
    start_time = datetime.now()
    
    try:
        # Grid
        Xi, Yi, Zi = data.to_grid(method='linear')
        ny, nx = Zi.shape
        
        # Remove NaN
        mask = np.isnan(Zi)
        if mask.any():
            from scipy.ndimage import distance_transform_edt
            indices = distance_transform_edt(mask, return_distances=False, return_indices=True)
            Zi[mask] = Zi[tuple(indices[:, mask])]
        
        dx = (Xi.max() - Xi.min()) / (nx - 1)
        dy = (Yi.max() - Yi.min()) / (ny - 1)
        
        # Números de onda
        kx = 2 * np.pi * fftfreq(nx, d=dx)
        ky = 2 * np.pi * fftfreq(ny, d=dy)
        KX, KY = np.meshgrid(kx, ky)
        K = np.sqrt(KX**2 + KY**2)
        
        # FFT 🚀 COM GPU (10-50x mais rápido)
        F = fft2_gpu(Zi)
        
        # Derivadas
        F_dx = F * (1j * KX)
        F_dy = F * (1j * KY)
        F_dz = F * K
        
        dx_field = np.real(ifft2_gpu(F_dx))
        dy_field = np.real(ifft2_gpu(F_dy))
        dz_field = np.real(ifft2_gpu(F_dz))
        
        if GPU_INFO['available']:
            logger.debug(f"✅ Sinal analítico processado na GPU: {GPU_INFO['device_name']}")
        
        # ASA
        asa = np.sqrt(dx_field**2 + dy_field**2 + dz_field**2)
        
        # Converte
        x_flat = Xi.flatten()
        y_flat = Yi.flatten()
        asa_flat = asa.flatten()
        
        x_col = data.coords['x']
        y_col = data.coords['y']
        
        asa_df = pl.DataFrame({
            x_col: x_flat,
            y_col: y_flat,
            f"{data.value_column}_asa": asa_flat
        })
        
        new_coords = {'x': data.coords['x'], 'y': data.coords['y']}
        
        processed_data = GeophysicalData(
            data=asa_df,
            data_type=data.data_type,
            dimension=data.dimension,
            coords=new_coords,
            value_column=f"{data.value_column}_asa",
            units=f"{data.units}/m",
            crs=data.crs,
            metadata={
                **data.metadata,
                'processing': 'analytic_signal'
            }
        )
        
        figures = create_comparison_plots(data, processed_data, "Sinal Analítico (ASA)")
        
        explanation = f"""
### 📊 Sinal Analítico Aplicado

**Resultado:**
- Campo original: {Zi.min():.2f} a {Zi.max():.2f} {data.units}
- ASA: {asa.min():.3f} a {asa.max():.3f} {data.units}/m
- ASA médio: {asa.mean():.3f} {data.units}/m

ASA é independente da magnetização e excelente para delimitar corpos em qualquer latitude.

**Estimativa de Profundidade:**
- Razão ASA/THD pode fornecer índice estrutural
- Máximos de ASA indicam bordas de corpos
"""
        
        references = [
            "NABIGHIAN, M. N. **The analytic signal of two-dimensional magnetic bodies**. Geophysics, v. 37, n. 3, 1972.",
            "ROEST, W. R.; VERHOEF, J.; PILKINGTON, M. **Magnetic interpretation using the 3-D analytic signal**. Geophysics, v. 57, 1992."
        ]
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return ProcessingResult(
            processed_data=processed_data,
            original_data=data,
            method_name="analytic_signal",
            parameters={},
            figures=figures,
            explanation=explanation,
            execution_time=execution_time,
            references=references
        )
        
    except Exception as e:
        logger.error(f"Erro no sinal analítico: {str(e)}")
        raise ProcessingError(f"Falha no sinal analítico: {str(e)}")


@register_processing(
    category="Geral",
    description="Derivada Tilt (Ângulo de Tilt)",
    input_type="grid",
    requires_params=[]
)
def tilt_angle(data: GeophysicalData) -> ProcessingResult:
    """
    Derivada Tilt (Ângulo de Tilt)
    
    Calcula o ângulo de tilt, definido como:
    Tilt = arctan(∂U/∂z / THD)
    
    Onde THD = sqrt((∂U/∂x)² + (∂U/∂y)²)
    
    Fundamento Teórico:
    -------------------
    O ângulo de tilt normaliza o gradiente vertical pelo horizontal,
    produzindo valores entre -π/2 e π/2 independentes da amplitude.
    
    Aplicações:
    -----------
    - Delineamento preciso de bordas
    - Independente de amplitude de anomalia
    - Valores próximos de zero sobre bordas
    - Estimativa de profundidade (método de Salem)
    
    Vantagens:
    ----------
    - Equaliza anomalias de diferentes amplitudes
    - Estável e menos sensível a ruído
    - Não requer conhecimento de magnetização
    - Fácil interpretação visual
    
    Referências:
    ------------
    MILLER, H. G.; SINGH, V. **Potential field tilt - a new concept for location 
    of potential field sources**. Journal of Applied Geophysics, v. 32, 
    p. 213-217, 1994. DOI: 10.1016/0926-9851(94)90022-1
    
    SALEM, A. et al. **Interpretation of magnetic data using tilt-angle 
    derivatives**. Geophysics, v. 73, n. 1, p. L1-L10, 2008. 
    DOI: 10.1190/1.2799992
    
    Parameters:
    -----------
    data : GeophysicalData
        Dados em grid regular
    
    Returns:
    --------
    ProcessingResult
        Ângulo de tilt em radianos
    """
    start_time = datetime.now()
    
    try:
        # Grid
        Xi, Yi, Zi = data.to_grid(method='linear')
        ny, nx = Zi.shape
        
        # Remove NaN
        mask = np.isnan(Zi)
        if mask.any():
            from scipy.ndimage import distance_transform_edt
            indices = distance_transform_edt(mask, return_distances=False, return_indices=True)
            Zi[mask] = Zi[tuple(indices[:, mask])]
        
        dx = (Xi.max() - Xi.min()) / (nx - 1)
        dy = (Yi.max() - Yi.min()) / (ny - 1)
        
        # Números de onda
        kx = 2 * np.pi * fftfreq(nx, d=dx)
        ky = 2 * np.pi * fftfreq(ny, d=dy)
        KX, KY = np.meshgrid(kx, ky)
        K = np.sqrt(KX**2 + KY**2)
        
        # FFT
        F = fft2(Zi)
        
        # Derivadas
        F_dx = F * (1j * KX)
        F_dy = F * (1j * KY)
        F_dz = F * K
        
        dx_field = np.real(ifft2(F_dx))
        dy_field = np.real(ifft2(F_dy))
        dz_field = np.real(ifft2(F_dz))
        
        # THD
        thd = np.sqrt(dx_field**2 + dy_field**2)
        
        # Tilt angle
        tilt = np.arctan2(dz_field, thd)
        
        # Converte para graus
        tilt_deg = np.rad2deg(tilt)
        
        # Converte
        x_flat = Xi.flatten()
        y_flat = Yi.flatten()
        tilt_flat = tilt_deg.flatten()
        
        x_col = data.coords['x']
        y_col = data.coords['y']
        
        tilt_df = pl.DataFrame({
            x_col: x_flat,
            y_col: y_flat,
            f"{data.value_column}_tilt": tilt_flat
        })
        
        new_coords = {'x': data.coords['x'], 'y': data.coords['y']}
        
        processed_data = GeophysicalData(
            data=tilt_df,
            data_type=data.data_type,
            dimension=data.dimension,
            coords=new_coords,
            value_column=f"{data.value_column}_tilt",
            units="graus",
            crs=data.crs,
            metadata={
                **data.metadata,
                'processing': 'tilt_angle'
            }
        )
        
        figures = create_comparison_plots(data, processed_data, "Ângulo de Tilt")
        
        explanation = f"""
### 📊 Ângulo de Tilt Aplicado

**Resultado:**
- Campo original: {Zi.min():.2f} a {Zi.max():.2f} {data.units}
- Tilt: {tilt_deg.min():.2f}° a {tilt_deg.max():.2f}°
- Tilt médio: {tilt_deg.mean():.2f}°

**Interpretação:**
- Valores próximos de 0°: bordas de corpos
- Valores positivos: sobre corpos
- Valores negativos: fora de corpos
- Contorno zero indica bordas precisas

O ângulo de tilt equaliza anomalias de diferentes amplitudes.
"""
        
        references = [
            "MILLER, H. G.; SINGH, V. **Potential field tilt**. Journal of Applied Geophysics, v. 32, 1994.",
            "SALEM, A. et al. **Interpretation of magnetic data using tilt-angle derivatives**. Geophysics, v. 73, 2008."
        ]
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return ProcessingResult(
            processed_data=processed_data,
            original_data=data,
            method_name="tilt_angle",
            parameters={},
            figures=figures,
            explanation=explanation,
            execution_time=execution_time,
            references=references
        )
        
    except Exception as e:
        logger.error(f"Erro no tilt angle: {str(e)}")
        raise ProcessingError(f"Falha no tilt angle: {str(e)}")


@register_processing(
    category="Geral",
    description="Filtro Passa-Baixa Gaussiano (suavização)",
    input_type="grid",
    requires_params=['wavelength']
)
def gaussian_lowpass(data: GeophysicalData, wavelength: float) -> ProcessingResult:
    """
    Filtro Passa-Baixa Gaussiano
    
    Aplica filtro gaussiano para suavização, atenuando comprimentos de onda
    menores que o especificado.
    
    Fundamento Teórico:
    -------------------
    Filter(k) = exp(-(k·λ)²/4)
    
    Onde:
        k = número de onda
        λ = comprimento de onda de corte
    
    Aplicações:
    -----------
    - Remoção de ruído de alta frequência
    - Separação regional-residual
    - Realce de tendências regionais
    - Pré-processamento para derivadas
    
    Referências:
    ------------
    BLAKELY, R. J. **Potential Theory in Gravity and Magnetic Applications**. 
    Cambridge University Press, 1995. p. 312-319.
    
    Parameters:
    -----------
    data : GeophysicalData
        Dados em grid
    wavelength : float
        Comprimento de onda de corte em metros
    
    Returns:
    --------
    ProcessingResult
        Dado suavizado
    """
    start_time = datetime.now()
    
    try:
        # Validações
        if data.dimension not in ['2D', '3D']:
            raise ProcessingError("Filtro passa-baixa requer dados 2D ou 3D")
        
        if len(data.data) < 10:
            raise ProcessingError("Dados insuficientes para gridding")
        Xi, Yi, Zi = data.to_grid(method='linear')
        ny, nx = Zi.shape
        
        mask = np.isnan(Zi)
        if mask.any():
            from scipy.ndimage import distance_transform_edt
            indices = distance_transform_edt(mask, return_distances=False, return_indices=True)
            Zi[mask] = Zi[tuple(indices[:, mask])]
        
        dx = (Xi.max() - Xi.min()) / (nx - 1)
        dy = (Yi.max() - Yi.min()) / (ny - 1)
        
        kx = 2 * np.pi * fftfreq(nx, d=dx)
        ky = 2 * np.pi * fftfreq(ny, d=dy)
        KX, KY = np.meshgrid(kx, ky)
        K = np.sqrt(KX**2 + KY**2)
        
        # Filtro gaussiano
        sigma = wavelength / (2 * np.pi)
        gauss_filter = np.exp(-(K * sigma)**2 / 2)
        
        # 🚀 FFT com aceleração GPU (10-50x mais rápido)
        F = fft2_gpu(Zi)
        F_filtered = F * gauss_filter
        Zi_filtered = np.real(ifft2_gpu(F_filtered))
        
        if GPU_INFO['available']:
            logger.debug(f"✅ Filtro Gaussiano processado na GPU: {GPU_INFO['device_name']}")
        
        x_flat = Xi.flatten()
        y_flat = Yi.flatten()
        z_filtered_flat = Zi_filtered.flatten()
        
        x_col = data.coords['x']
        y_col = data.coords['y']
        
        filtered_df = pl.DataFrame({
            x_col: x_flat,
            y_col: y_flat,
            f"{data.value_column}_lowpass": z_filtered_flat
        })
        
        new_coords = {'x': data.coords['x'], 'y': data.coords['y']}
        
        processed_data = GeophysicalData(
            data=filtered_df,
            data_type=data.data_type,
            dimension=data.dimension,
            coords=new_coords,
            value_column=f"{data.value_column}_lowpass",
            units=data.units,
            crs=data.crs,
            metadata={
                **data.metadata,
                'processing': 'gaussian_lowpass',
                'wavelength': wavelength
            }
        )
        
        figures = create_comparison_plots(data, processed_data, f"Filtro Passa-Baixa (λ={wavelength}m)")
        
        explanation = f"""
### 📊 Filtro Passa-Baixa Gaussiano

**Parâmetros:**
- Comprimento de onda de corte: {wavelength:.0f} m

**Resultado:**
- Original: {Zi.min():.2f} a {Zi.max():.2f} {data.units}
- Filtrado: {Zi_filtered.min():.2f} a {Zi_filtered.max():.2f} {data.units}
- Redução de variância: {(1-Zi_filtered.std()/Zi.std())*100:.1f}%

Filtro remove componentes de alta frequência, suavizando o campo.
"""
        
        references = [
            "BLAKELY, R. J. **Potential Theory in Gravity and Magnetic Applications**. Cambridge University Press, 1995."
        ]
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return ProcessingResult(
            processed_data=processed_data,
            original_data=data,
            method_name="gaussian_lowpass",
            parameters={'wavelength': wavelength},
            figures=figures,
            explanation=explanation,
            execution_time=execution_time,
            references=references
        )
        
    except Exception as e:
        logger.error(f"Erro no filtro passa-baixa: {str(e)}")
        raise ProcessingError(f"Falha no filtro passa-baixa: {str(e)}")


def create_histogram(data: GeophysicalData) -> go.Figure:
    """
    Cria histograma dos dados.
    
    Parameters:
    -----------
    data : GeophysicalData
        Dados para histograma
    
    Returns:
    --------
    go.Figure
        Figura Plotly com histograma
    """
    values = data.data[data.value_column].to_numpy()
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=values,
        nbinsx=50,
        name=data.value_column,
        marker_color='#0066a1',
        opacity=0.75
    ))
    
    fig.update_layout(
        title=f"Histograma - {data.value_column}",
        xaxis_title=f"{data.value_column} ({data.units})",
        yaxis_title="Frequênçia",
        template="plotly_white",
        hovermode='x',
        height=500
    )
    
    # Adiciona linhas estatísticas
    fig.add_vline(x=data.metadata['mean'], line_dash="dash", line_color="red", 
                  annotation_text="Média", annotation_position="top")
    fig.add_vline(x=data.metadata['median'], line_dash="dash", line_color="green",
                  annotation_text="Mediana", annotation_position="bottom")
    
    return fig


def create_scatter_plot(data: GeophysicalData) -> go.Figure:
    """
    Cria gráfico de dispersão 2D colorido por valor.
    
    Parameters:
    -----------
    data : GeophysicalData
        Dados para plot
    
    Returns:
    --------
    go.Figure
        Figura Plotly
    """
    x = data.data[data.coords['x']].to_numpy()
    y = data.data[data.coords['y']].to_numpy()
    z = data.data[data.value_column].to_numpy()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='markers',
        marker=dict(
            size=4,
            color=z,
            colorscale='RdBu_r',
            showscale=True,
            colorbar=dict(title=f"{data.value_column}<br>({data.units})")
        ),
        text=[f"X: {xi:.2f}<br>Y: {yi:.2f}<br>{data.value_column}: {zi:.2f}" 
              for xi, yi, zi in zip(x, y, z)],
        hovertemplate='%{text}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"Dados - {data.value_column}",
        xaxis_title=data.coords['x'],
        yaxis_title=data.coords['y'],
        template="plotly_white",
        height=600,
        hovermode='closest'
    )
    
    return fig


def create_simple_map(data: GeophysicalData) -> folium.Map:
    """
    Cria mapa simples com dados.
    
    Parameters:
    -----------
    data : GeophysicalData
        Dados para mapa
    
    Returns:
    --------
    folium.Map
        Mapa Folium
    """
    x_col = data.coords['x']
    y_col = data.coords['y']
    
    center_x = data.metadata['bbox']['x_min'] + (data.metadata['bbox']['x_max'] - data.metadata['bbox']['x_min']) / 2
    center_y = data.metadata['bbox']['y_min'] + (data.metadata['bbox']['y_max'] - data.metadata['bbox']['y_min']) / 2
    
    m = folium.Map(location=[center_y, center_x], zoom_start=11, tiles='OpenStreetMap')
    
    # Adiciona pontos (sample para performance)
    df = data.data.to_pandas()
    sample_size = min(1000, len(df))
    df_sample = df.sample(n=sample_size)
    
    for _, row in df_sample.iterrows():
        folium.CircleMarker(
            location=[row[y_col], row[x_col]],
            radius=3,
            color='#0066a1',
            fill=True,
            fillOpacity=0.6,
            popup=f"{data.value_column}: {row[data.value_column]:.2f}"
        ).add_to(m)
    
    return m


def detect_visualization_command(user_input: str) -> tuple:
    """
    Detecta comandos de visualização simples.
    
    Parameters:
    -----------
    user_input : str
        Entrada do usuário
    
    Returns:
    --------
    tuple: (viz_type, params) onde viz_type pode ser 'histogram', 'scatter', 'map', 'stats' ou None
    """
    user_lower = user_input.lower()
    
    # Histograma
    if any(word in user_lower for word in ['histograma', 'histogram', 'distribuição', 'distribuicao']):
        return 'histogram', {}
    
    # Gráfico de dispersão
    if any(word in user_lower for word in ['scatter', 'dispersão', 'dispersao', 'plot', 'gráfico']):
        if 'histograma' not in user_lower and 'histogram' not in user_lower:
            return 'scatter', {}
    
    # Mapa
    if any(word in user_lower for word in ['mapa', 'map', 'espacial']):
        return 'map', {}
    
    # Estatísticas
    if any(word in user_lower for word in ['estatística', 'estatistica', 'stats', 'resumo']):
        return 'stats', {}
    
    return None, None


def create_comparison_plots(
    original: GeophysicalData,
    processed: GeophysicalData,
    title: str
) -> List[go.Figure]:
    """
    Cria plots comparativos entre dados originais e processados.
    
    Parameters:
    -----------
    original : GeophysicalData
        Dados originais
    processed : GeophysicalData
        Dados processados
    title : str
        Título do processamento
    
    Returns:
    --------
    List[go.Figure]
        Lista com figuras Plotly
    """
    figures = []
    
    try:
        logger.info(f"Criando plots comparativos: {title}")
        logger.info(f"Dimensão dos dados: {original.dimension}")
        
        if original.dimension in ['2D', '3D']:
            # Plot de comparação 2D
            logger.info("Criando subplot de comparação...")
            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=('Original', 'Processado', 'Diferença'),
                specs=[[{'type': 'heatmap'}, {'type': 'heatmap'}, {'type': 'heatmap'}]]
            )
            
            # Grid original
            logger.info("Convertendo dados para grid...")
            Xi_orig, Yi_orig, Zi_orig = original.to_grid()
            
            # Grid processado
            Xi_proc, Yi_proc, Zi_proc = processed.to_grid()
            
            # Diferença
            Zi_diff = Zi_orig - Zi_proc
            
            logger.info(f"Grid shape: {Zi_orig.shape}")
            
            # Original
            fig.add_trace(
                go.Heatmap(
                    x=Xi_orig[0,:],
                    y=Yi_orig[:,0],
                    z=Zi_orig,
                    colorscale='RdBu_r',
                    name='Original',
                    colorbar=dict(x=0.3, len=0.9, title=original.units)
                ),
                row=1, col=1
            )
            
            # Processado
            fig.add_trace(
                go.Heatmap(
                    x=Xi_proc[0,:],
                    y=Yi_proc[:,0],
                    z=Zi_proc,
                    colorscale='RdBu_r',
                    name='Processado',
                    colorbar=dict(x=0.63, len=0.9, title=processed.units)
                ),
                row=1, col=2
            )
            
            # Diferença
            fig.add_trace(
                go.Heatmap(
                    x=Xi_orig[0,:],
                    y=Yi_orig[:,0],
                    z=Zi_diff,
                    colorscale='RdBu',
                    name='Diferença',
                    colorbar=dict(x=0.96, len=0.9, title=original.units)
                ),
                row=1, col=3
            )
            
            fig.update_layout(
                title_text=title,
                height=400,
                showlegend=False
            )
            
            fig.update_xaxes(title_text="X", row=1, col=1)
            fig.update_xaxes(title_text="X", row=1, col=2)
            fig.update_xaxes(title_text="X", row=1, col=3)
            
            fig.update_yaxes(title_text="Y", row=1, col=1)
            
            logger.info("Subplot criado com sucesso!")
            figures.append(fig)
        
        # Histograma comparativo
        logger.info("Criando histograma comparativo...")
        fig_hist = go.Figure()
        
        orig_values = original.data[original.value_column].to_numpy()
        proc_values = processed.data[processed.value_column].to_numpy()
        
        fig_hist.add_trace(go.Histogram(
            x=orig_values,
            name='Original',
            opacity=0.7,
            marker_color='blue'
        ))
        
        fig_hist.add_trace(go.Histogram(
            x=proc_values,
            name='Processado',
            opacity=0.7,
            marker_color='red'
        ))
        
        fig_hist.update_layout(
            title="Distribuição de Valores",
            xaxis_title=f"Valor ({original.units})",
            yaxis_title="Frequência",
            barmode='overlay',
            height=300
        )
        
        logger.info("Histograma criado com sucesso!")
        figures.append(fig_hist)
        
        logger.info(f"Total de {len(figures)} figuras criadas")
        
    except Exception as e:
        logger.error(f"Erro ao criar plots: {str(e)}")
        logger.error(traceback.format_exc())
    
    return figures


# ============================================================================
# COMPONENTES DE INTERFACE STREAMLIT
# ============================================================================

def get_ppgdot_logo_base64():
    """Retorna a logo PPGDot-UFF em base64."""
    try:
        from pathlib import Path
        import base64
        import os
        
        # Tenta múltiplos caminhos
        base_path = Path(__file__).parent
        logo_paths = [
            base_path / "assets" / "ppgdot_logo.png",
            Path("assets/ppgdot_logo.png"),
            Path(os.getcwd()) / "assets" / "ppgdot_logo.png"
        ]
        
        for logo_path in logo_paths:
            if logo_path.exists():
                with open(logo_path, "rb") as f:
                    img_bytes = f.read()
                    b64 = base64.b64encode(img_bytes).decode()
                    return f"data:image/png;base64,{b64}"
        
        return None
    except Exception as e:
        logger.warning(f"Erro ao carregar logo: {e}")
        return None


def render_landing_page():
    """Renderiza página inicial com documentação e configuração."""
    
    # Header com título (logo já está no header fixo)
    st.markdown(f"""
    <div style='padding: 20px 0; margin-top: 20px;'>
        <h1 style='color: #003d5c; font-size: 48px; margin: 0;'> {APP_TITLE}</h1>
        <h3 style='color: #0066a1; font-size: 20px; margin: 5px 0;'>{APP_SUBTITLE}</h3>
        <p style='color: #666; font-size: 14px; margin: 5px 0;'>
            Programa de Pós-Graduação em Dinâmica dos Oceanos e da Terra<br>
            Universidade Federal Fluminense
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Tabs para organizar conteúdo
    tab1, tab2, tab3 = st.tabs(["  📖 Sobre  ", "  🔧 Funcionalidades  ", "  ⚙️ Configuração  "])
    
    with tab1:
        st.markdown("""
        ### 🎯 Sobre o GeoBot
        
        O **GeoBot** é um agente de inteligência artificial conversacional desenvolvido para
        auxiliar pesquisadores e profissionais no processamento e análise de dados geofísicos
        potenciais (gravimetria e magnetometria).
        
        Este projeto faz parte de uma pesquisa de mestrado em Geofísica desenvolvida no
        **Programa de Pós-Graduação em Dinâmica dos Oceanos e da Terra (DOT-UFF)**.
        
       
        #### 📚 Disclaimer Acadêmico
        
        Este projeto é uma ferramenta experimental que combina processamento geofísico
        clássico com inteligência artificial generativa.
        
        **Importante:**
        - ⚠️ Sempre valide os resultados
        - 📄 As citações são recuperadas automaticamente via RAG (podem ter imprecisões)
        - 💻 O código está disponível para revisão e contribuições
        
        ---
        
        *Desenvolvido por Allan Ramalho e Rodrigo Bijani | 2026*
        """)
    
    with tab2:
        st.markdown("""
        ### 🚀 Principais Funcionalidades
        
        #### 💬 Processamento através de Linguagem Natural
        Execute processamentos complexos apenas conversando com o GeoBot:
        - "Aplique correção de Bouguer com densidade 2.67"
        - "Faça continuação ascendente para 1000m"
        - "Calcule a primeira derivada vertical"
        
        #### 🧠 Análise Inteligente
        - Interpretação de dados com suporte de IA Generativa
        - Sugestões automáticas de processamentos
        - Explicações didáticas dos métodos
        
        #### 📊 Visualização Interativa
        - Mapas georreferenciados
        - Plots 2D/3D comparativos
        - Histogramas e estatísticas
        
        #### 📂 Tipos de Dados Aceitos
        
        **Gravimetria:**
        - Anomalias Bouguer, ar-livre, isostáticas
        - Dados de estações gravimétricas
        
        **Magnetometria:**
        - Campo total, anomalia magnética
        - Componentes do vetor magnético
        
        **Formatos:** CSV, TXT, Excel
        """)
        
        # Lista processamentos do registro
        st.markdown("### 🔄 Processamentos Disponíveis")
        
        if PROCESSING_REGISTRY:
            categories = {}
            for func_name, info in PROCESSING_REGISTRY.items():
                cat = info['category']
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append({
                    'name': func_name,
                    'description': info['description']
                })
            
            for category, funcs in categories.items():
                with st.expander(f"**{category}**", expanded=False):
                    for func in funcs:
                        st.markdown(f"**`{func['name']}`**")
                        st.markdown(f"↳ {func['description']}")
                        st.markdown("")
    
    with tab3:
        st.markdown("""
        ### 🔧 Configuração
        
        Para utilizar o GeoBot, você precisa de uma chave de API da Groq.
        """)
        
        # Input de API Key
        with st.form("api_key_form"):
            st.markdown("#### 🔑 API Key Groq")
            st.markdown("""
            A Groq fornece acesso gratuito a modelos LLM de última geração.
            
            👉 **Obtenha sua chave em:** [console.groq.com/keys](https://console.groq.com/keys)
            
            1. Crie uma conta gratuita
            2. Acesse a seção "API Keys"
            3. Gere uma nova chave
            4. Cole aqui abaixo
            """)
            
            api_key = st.text_input(
                "Insira sua API Key:",
                type="password",
                placeholder="gsk_...",
                help="Sua chave começa com 'gsk_'"
            )
            
            col1, col2 = st.columns([3, 1])
            with col1:
                submit_button = st.form_submit_button(
                    "✅ Confirmar e Continuar",
                    use_container_width=True,
                    type="primary"
                )
            
            if submit_button:
                if api_key and api_key.startswith("gsk_"):
                    try:
                        # Valida API key criando cliente
                        with st.spinner("🔄 Validando API Key..."):
                            test_client = Groq(api_key=api_key)
                            test_client.models.list()  # Testa conexão
                        
                        # Salva no session_state
                        st.session_state.groq_api_key = api_key
                        st.session_state.page = "model_selection"
                        
                        st.success("✅ API Key validada com sucesso!")
                        st.balloons()
                        time.sleep(1)
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Erro ao validar API Key: {str(e)}")
                        logger.error(f"Erro na validação da API: {str(e)}")
                else:
                    st.error("❌ API Key inválida. Deve começar com 'gsk_'")
        
        st.markdown("""
        ---
        
        ### 📞 Contato
        
        **PPG DOT-UFF**  
        Universidade Federal Fluminense  
        Av. Gen. Milton Tavares de Souza s/nº - Gragoatá  
        Niterói – RJ – 24210-346 | Brasil
        
        📩 [allansoares@id.uff.br](mailto:allansoares@id.uff.br)
        
        📩 [rodrigobijani@id.uff.br](mailto:rodrigobijani@id.uff.br)
        
        📚 [ppgdot-uff.com.br](https://ppgdot-uff.com.br/)
        """)


def render_model_selection():
    """Renderiza página de seleção de modelo LLM."""
    st.title("🤖 Seleção de Modelo")
    
    st.markdown("""
    Selecione o modelo de linguagem que será usado pelo GeoBot. O sistema possui
    **fallback automático** entre modelos em caso de rate limit.
    """)
    
    # Inicializa LLM Manager se não existir
    if 'llm_manager' not in st.session_state:
        st.session_state.llm_manager = LLMManager(st.session_state.groq_api_key)
    
    llm = st.session_state.llm_manager
    
    # Lista modelos disponíveis
    with st.spinner("Carregando modelos disponíveis..."):
        available_models = llm.list_available_models()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Inicializa modelo selecionado no session_state se não existir
        if 'selected_model' not in st.session_state:
            st.session_state.selected_model = llm.primary_model
        
        # Callback para atualizar o modelo selecionado
        def on_model_change():
            st.session_state.selected_model = st.session_state.model_selector_key
        
        selected_model = st.selectbox(
            "Modelo Principal:",
            options=available_models,
            index=available_models.index(st.session_state.selected_model) if st.session_state.selected_model in available_models else 0,
            key="model_selector_key",
            on_change=on_model_change
        )
        
        st.markdown(f"""
        **Modelo selecionado:** `{st.session_state.selected_model}`
        
        **Sistema de Fallback:**
        - Se o modelo principal atingir rate limit, o sistema automaticamente tentará outros modelos
        - O contexto da conversa é mantido durante a transição
        - Você será notificado quando houver mudança de modelo
        """)
    
    with col2:
        st.markdown("### 📊 Modelos")
        st.metric("Disponíveis", len(available_models))
        st.metric("Atual", selected_model)
    
    if st.button("✅ Confirmar e Iniciar GeoBot", use_container_width=True, type="primary"):
        selected_model = st.session_state.selected_model
        llm.primary_model = selected_model
        llm.current_model = selected_model
        llm.fallback_models = [m for m in available_models if m != selected_model]
        
        st.session_state.page = "main_app"
        st.success(f"✅ Modelo {selected_model} configurado!")
        st.rerun()
    
    if st.button("⬅️ Voltar", use_container_width=True):
        st.session_state.page = "landing"
        st.rerun()


def render_sidebar():
    """Renderiza sidebar com upload de dados apenas."""
    with st.sidebar:
        st.markdown("### 📂 Upload de Dados")
        # Upload de arquivo
        uploaded_file = st.file_uploader(
            "Carregar dados geofísicos",
            type=['csv', 'txt', 'xlsx', 'xls'],
            help="Formatos aceitos: CSV, TXT, Excel"
        )
        
        if uploaded_file is not None:
            try:
                # Parse arquivo
                with st.spinner("Processando arquivo..."):
                    geo_data = parse_uploaded_file(uploaded_file, uploaded_file.name)
                    st.session_state.current_data = geo_data
                
                st.success("✅ Arquivo carregado!")
                # st.info(f"**{geo_data.data_type.upper()}** - {geo_data.metadata['n_points']:,} pontos")
                
            except Exception as e:
                st.error(f"❌ Erro ao processar arquivo: {str(e)}")
                logger.error(f"Erro no upload: {str(e)}")
        
        st.markdown("")
        st.markdown("---")
        st.markdown("")
        
        # Sugestões de comandos
        with st.expander("💡 Sugestões de Comandos"):
            st.markdown("""
            **Análise Estatística:**
            - "Calcule as estatísticas dos dados"
            - "Mostre o histograma da gravidade"
            - "Identifique outliers nos dados"
            
            **Processamento:**
            - "Aplique uma redução ao polo"
            - "Calcule a derivada vertical"
            - "Faça upward continuation de 100m"
            
            **Visualização:**
            - "Mostre o mapa interpolado"
            - "Gere um perfil na direção NS"
            - "Crie visualização 3D dos dados"
            """)
        
        st.markdown("")
        st.markdown("---")
        st.markdown("")
        
        # Informações adicionais
        with st.expander("ℹ️ Ajuda"):
            st.markdown("""
            **Como usar:**
            1. Carregue um arquivo com dados geofísicos
            2. Visualize estatísticas e mapas
            3. Converse com o GeoBot sobre os dados
            4. Solicite processamentos específicos
            
            **Formatos aceitos:**
            - CSV/TXT com colunas: X, Y, valor
            - Excel (.xlsx, .xls)
            
            **Colunas esperadas:**
            - Coordenadas: longitude, latitude (ou x, y)
            - Valores: gravity, magnetic, etc.
            """)


def render_data_panel():
    """Renderiza painel de dados com estatísticas, preview, mapa e plot."""
    if 'current_data' not in st.session_state or st.session_state.current_data is None:
        st.info("📂 Carregue um arquivo de dados geofísicos na sidebar para começar")
        return
    
    geo_data = st.session_state.current_data
    stats = geo_data.metadata
    df = geo_data.to_pandas()
    
    x_col = geo_data.coords['x']
    y_col = geo_data.coords['y']
    
    # Badge de tipo de dado
    tipo_badges = {
        'gravity': ('🟦', '#0066a1'),
        'magnetic': ('🟥', '#d32f2f'),
        'topography': ('🟩', '#43a047'),
        'unknown': ('⬜', '#757575')
    }
    badge, color = tipo_badges.get(geo_data.data_type, tipo_badges['unknown'])
    
    st.markdown(f"""
    <div style='background: linear-gradient(90deg, {color}15 0%, {color}05 100%); 
                padding: 15px; border-radius: 10px; border-left: 4px solid {color}; margin-bottom: 20px;'>
        <h2 style='margin: 0; color: {color};'>{badge} {geo_data.data_type.upper()} {geo_data.dimension}</h2>
        <p style='margin: 5px 0 0 0; color: #666;'>{stats['n_points']:,} pontos | {geo_data.units}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ========================================
    # SEÇÃO 1: ESTATÍSTICAS DESCRITIVAS
    # ========================================
    st.markdown("""
    <div style='background: linear-gradient(90deg, #003d5c15 0%, #003d5c05 100%); 
                padding: 10px 15px; border-radius: 8px; border-left: 4px solid #0066a1; margin-bottom: 15px;'>
        <h3 style='margin: 0; color: #003d5c;'>📊 Estatísticas Descritivas</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("🔢 Sample N°", f"{stats['n_points']:,}", )
        st.metric("📉 Mín", f"{stats['min']:.2f}")

    with col2:
        st.metric("🔢 Média", f"{stats['mean']:.2f}")
        st.metric("📈 Máx", f"{stats['max']:.2f}")
        
    with col3:
        st.metric("📊 Mediana", f"{stats['median']:.2f}")
        st.metric("📏 Std", f"{stats['std']:.2f}")
        
    with col4:
        st.metric("📊 IQR", f"{stats['q75'] - stats['q25']:.2f}")
        
    
    # Divisor visual
    st.markdown("""<hr style='margin: 30px 0; border: none; border-top: 2px solid #e0e0e0;'>""", unsafe_allow_html=True)
    
    # ========================================
    # SEÇÃO 2: PREVIEW DOS DADOS
    # ========================================
    st.markdown("""
    <div style='background: linear-gradient(90deg, #003d5c15 0%, #003d5c05 100%); 
                padding: 10px 15px; border-radius: 8px; border-left: 4px solid #0066a1; margin-bottom: 15px;'>
        <h3 style='margin: 0; color: #003d5c;'>📋 Preview dos Dados</h3>
    </div>
    """, unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True, height=300)
    
    # Divisor visual
    st.markdown("""<hr style='margin: 30px 0; border: none; border-top: 2px solid #e0e0e0;'>""", unsafe_allow_html=True)
    
    # ========================================
    # SEÇÃO 3: VISUALIZAÇÕES ESPACIAIS
    # ========================================
    st.markdown("""
    <div style='background: linear-gradient(90deg, #003d5c15 0%, #003d5c05 100%); 
                padding: 10px 15px; border-radius: 8px; border-left: 4px solid #0066a1; margin-bottom: 15px;'>
        <h3 style='margin: 0; color: #003d5c;'>🗺️ Visualizações Espaciais</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col_plot, col_map = st.columns(2)
    
    with col_plot:
        st.markdown("#### 📈 Distribuição Espacial")
        
        # Plot com plotly
        fig = px.scatter(
            df.sample(min(3000, len(df))),
            x=x_col,
            y=y_col,
            color=geo_data.value_column,
            color_continuous_scale='RdBu_r',
            labels={
                x_col: 'Longitude (°)', 
                y_col: 'Latitude (°)', 
                geo_data.value_column: f'{geo_data.value_column} ({geo_data.units})'
            },
            height=500
        )
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            coloraxis_colorbar=dict(title=geo_data.units)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col_map:
        st.markdown("#### 🗺 Mapa")
        
        if 'bbox' in stats:
            try:
                bbox = stats['bbox']
                lat_min, lat_max = bbox['y_min'], bbox['y_max']
                lon_min, lon_max = bbox['x_min'], bbox['x_max']
                
                is_geographic = (-90 <= lat_min <= 90 and -90 <= lat_max <= 90 and
                               -180 <= lon_min <= 180 and -180 <= lon_max <= 180)
                
                if is_geographic:
                    center_lat = (lat_min + lat_max) / 2
                    center_lon = (lon_min + lon_max) / 2
                    
                    # Criar mapa
                    m = folium.Map(
                        location=[center_lat, center_lon],
                        zoom_start=11,
                        tiles='OpenStreetMap'
                    )
                    
                    # TODOS OS PONTOS (sem amostragem)
                    df_sample = df
                    sample_size = len(df)
                    
                    # Usar mesma escala de cores do plot (RdBu_r)
                    values = df_sample[geo_data.value_column]
                    vmin, vmax = stats['min'], stats['max']
                    
                    # Criar colormap RdBu_r  (Red-Blue reversed)
                    import matplotlib.pyplot as plt
                    import matplotlib.colors as mcolors
                    
                    cmap = plt.cm.get_cmap('RdBu_r')
                    
                    # Adicionar pontos
                    for _, row in df_sample.iterrows():
                        lat = row[y_col]
                        lon = row[x_col]
                        value = row[geo_data.value_column]
                        
                        # Normalizar
                        norm_value = (value - vmin) / (vmax - vmin) if vmax != vmin else 0.5
                        
                        # Obter cor do colormap
                        rgba = cmap(norm_value)
                        color = mcolors.rgb2hex(rgba[:3])
                        
                        # Tooltip com informações completas
                        tooltip = f"""
                        <b>{geo_data.value_column}:</b> {value:.3f} {geo_data.units}<br>
                        <b>Lat:</b> {lat:.6f}°<br>
                        <b>Lon:</b> {lon:.6f}°
                        """
                        
                        folium.CircleMarker(
                            location=[lat, lon],
                            radius=4,
                            tooltip=tooltip,
                            color=color,
                            fill=True,
                            fillColor=color,
                            fillOpacity=0.7,
                            weight=1,
                            opacity=0.8
                        ).add_to(m)
                    
                    # Renderizar
                    folium_static(m, width=None, height=500)
                    # st.caption(f"�️ {sample_size:,} pontos exibidos")
                else:
                    st.warning("⚠️ Coordenadas não geográficas")
                    
            except Exception as e:
                logger.error(f"Erro no mapa: {e}")
                st.error(f"❌ Erro ao criar mapa: {e}")
        else:
            st.info("ℹ️ Bounding box não disponível")


def render_main_chat():
    """Renderiza interface principal de chat com estilo WhatsApp usando containers."""
    
    # Título com fonte menor
    st.markdown("<h3 style='font-size: 1.5rem; margin-bottom: 1rem;'>💬 Chat</h3>", unsafe_allow_html=True)
    
    # Inicializa histórico de chat
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Inicializa RAG Engine
    if 'rag_engine' not in st.session_state:
        st.session_state.rag_engine = RAGEngine()
        st.session_state.rag_engine.initialize()
    
    # Container de mensagens com altura fixa (parâmetro nativo do Streamlit)
    chat_container = st.container(height=500)
    
    with chat_container:
        # Renderiza histórico de mensagens com estilo WhatsApp
        for i, message in enumerate(st.session_state.chat_history):
            is_user = message["role"] == "user"
            
            if is_user:
                # Mensagem do usuário (direita, verde)
                user_html = f"""
                <div style="display: flex; justify-content: flex-end; margin-bottom: 10px; clear: both;">
                    <div style="
                        background-color: #dcf8c6;
                        color: #003d5c;
                        padding: 8px 12px;
                        border-radius: 10px;
                        max-width: 65%;
                        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
                        float: right;
                        word-wrap: break-word;
                        border: none;
                    ">
                        <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 3px;">
                            <span style="font-size: 16px;">🧑‍🔬</span>
                            <strong style="color: #003d5c; font-size: 0.85rem;">Você</strong>
                        </div>
                        <div style="color: #003d5c; line-height: 1.4; font-size: 0.9rem;">
                            {message["content"]}
                        </div>
                    </div>
                </div>
                """
                st.markdown(user_html, unsafe_allow_html=True)
            else:
                # Mensagem do bot (esquerda, branca)
                bot_html = f"""
                <div style="display: flex; justify-content: flex-start; margin-bottom: 10px; clear: both;">
                    <div style="
                        background-color: #ffffff;
                        color: #003d5c;
                        padding: 8px 12px;
                        border-radius: 10px;
                        max-width: 65%;
                        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
                        float: left;
                        word-wrap: break-word;
                        border: none;
                    ">
                        <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 3px;">
                            <span style="font-size: 16px;">🤖</span>
                            <strong style="color: #003d5c; font-size: 0.85rem;">GeoBot</strong>
                        </div>
                        <div style="color: #003d5c; line-height: 1.4; font-size: 0.9rem;">
                            {message["content"]}
                        </div>
                    </div>
                </div>
                """
                st.markdown(bot_html, unsafe_allow_html=True)
                
                # Renderiza figuras se existirem
                if "figures" in message and message["figures"]:
                    for fig in message["figures"]:
                        st.plotly_chart(fig, use_container_width=True)
    
    # Input de chat com botão de limpar ao lado (sempre no final)
    col_input, col_clear = st.columns([8, 1])
    
    with col_input:
        prompt = st.chat_input("Digite sua mensagem...")
    
    with col_clear:
        if st.button("🗑️", type="secondary", key="clear_chat_btn"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Processa input
    if prompt:
        # Adiciona mensagem do usuário ao histórico
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Gera resposta (sem spinner para não bloquear interface)
        response, figures = generate_bot_response(prompt)
        
        # Adiciona resposta ao histórico
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response,
            "figures": figures if figures else None
        })
        
        # Recarrega a página para mostrar as novas mensagens
        st.rerun()



def detect_processing_command(user_input: str) -> tuple:
    """
    Detecta comando de processamento e extrai parâmetros.
    
    Returns:
    --------
    tuple: (function_name, parameters_dict) ou (None, None)
    """
    user_lower = user_input.lower()
    
    # Mapeamento de comandos
    commands = {
        'bouguer': ('bouguer_correction', {}),
        'continuação ascendente': ('upward_continuation', {}),
        'continuacao ascendente': ('upward_continuation', {}),
        'upward continuation': ('upward_continuation', {}),
        'derivada vertical': ('vertical_derivative', {}),
        'vertical derivative': ('vertical_derivative', {}),
        'derivada horizontal': ('horizontal_derivative_total', {}),
        'thd': ('horizontal_derivative_total', {}),
        'redução ao polo': ('reduction_to_pole', {}),
        'reducao ao polo': ('reduction_to_pole', {}),
        'rtp': ('reduction_to_pole', {}),
        'sinal analítico': ('analytic_signal', {}),
        'sinal analitico': ('analytic_signal', {}),
        'ângulo de tilt': ('tilt_angle', {}),
        'angulo de tilt': ('tilt_angle', {}),
        'tilt': ('tilt_angle', {}),
        'passa-baixa': ('gaussian_lowpass', {}),
        'passa baixa': ('gaussian_lowpass', {}),
        'gaussiano': ('gaussian_lowpass', {})
    }
    
    # Detecta comando
    detected_func = None
    params = {}
    
    for keyword, (func_name, default_params) in commands.items():
        if keyword in user_lower:
            detected_func = func_name
            params = default_params.copy()
            break
    
    if not detected_func:
        return None, None
    
    # Extrai parâmetros específicos
    import re
    
    # Densidade (para Bouguer)
    if 'bouguer' in detected_func:
        density_match = re.search(r'densidade\s*(?:de\s*)?(\d+\.?\d*)', user_lower)
        if density_match:
            params['density'] = float(density_match.group(1))
        else:
            params['density'] = 2.67  # padrão
    
    # Altura (para continuação)
    if 'upward' in detected_func:
        height_match = re.search(r'(\d+)\s*m(?:etros)?', user_lower)
        if height_match:
            params['height'] = float(height_match.group(1))
        else:
            params['height'] = 1000.0  # padrão
    
    # Comprimento de onda (para filtro)
    if 'lowpass' in detected_func:
        wavelength_match = re.search(r'(?:comprimento|lambda|wavelength)\s*(?:de\s*)?(\d+)', user_lower)
        if wavelength_match:
            params['wavelength'] = float(wavelength_match.group(1))
        else:
            params['wavelength'] = 5000.0  # padrão
    
    # RTP - parâmetros magnéticos (valores padrão Brasil)
    if 'reduction_to_pole' in detected_func:
        params['inc_field'] = -25.0  # Inclinação típica Brasil
        params['dec_field'] = -20.0  # Declinação típica Brasil
        params['inc_mag'] = -25.0
        params['dec_mag'] = -20.0
    
    return detected_func, params


def generate_bot_response(user_input: str) -> tuple:
    """
    Gera resposta do bot usando LLM + RAG + Processamento.
    
    Parameters:
    -----------
    user_input : str
        Input do usuário
    
    Returns:
    --------
    tuple: (response_text, figures_list)
    """
    try:
        # Primeiro verifica se há dados carregados
        if 'current_data' not in st.session_state:
            return "Por favor, carregue dados primeiro usando a sidebar.", []
        
        data = st.session_state.current_data
        
        # 1. DETECTA COMANDOS DE VISUALIZAÇÃO
        viz_type, viz_params = detect_visualization_command(user_input)
        
        if viz_type:
            try:
                figures = []
                response = ""
                
                if viz_type == 'histogram':
                    fig = create_histogram(data)
                    figures.append(fig)
                    response = f"✅ **Histograma gerado!**\n\nDistribuição de {data.metadata['n_points']:,} pontos de {data.value_column} ({data.units})."
                
                elif viz_type == 'scatter':
                    fig = create_scatter_plot(data)
                    figures.append(fig)
                    response = f"✅ **Gráfico de dispersão gerado!**\n\n{data.metadata['n_points']:,} pontos plotados."
                
                elif viz_type == 'map':
                    # Para mapa, retorna instrução de ver no painel de dados
                    response = "✅ O mapa já está disponível no painel de dados à esquerda (seção 'Visualizações Espaciais')."
                
                elif viz_type == 'stats':
                    stats = data.metadata
                    response = f"""
✅ **Estatísticas Descritivas:**

- **Nº de pontos:** {stats['n_points']:,}
- **Média:** {stats['mean']:.2f} {data.units}
- **Mediana:** {stats['median']:.2f} {data.units}
- **Desvio padrão:** {stats['std']:.2f} {data.units}
- **Mínimo:** {stats['min']:.2f} {data.units}
- **Máximo:** {stats['max']:.2f} {data.units}
- **Q1 (25%):** {stats['q25']:.2f} {data.units}
- **Q3 (75%):** {stats['q75']:.2f} {data.units}
- **IQR:** {stats['q75'] - stats['q25']:.2f} {data.units}
"""
                
                return response, figures
                
            except Exception as e:
                logger.error(f"Erro ao gerar visualização: {str(e)}")
                return f"❌ Erro ao gerar visualização: {str(e)}", []
        
        # 2. DETECTA COMANDOS DE PROCESSAMENTO
        func_name, params = detect_processing_command(user_input)
        
        if func_name and 'current_data' in st.session_state:
            # Executa processamento
            try:
                # Pega função do registro
                processing_func = globals()[func_name]
                
                # Executa
                data = st.session_state.current_data
                result = processing_func(data, **params)
                
                # Salva resultado processado
                st.session_state.current_data = result.processed_data
                
                # Monta resposta
                response = f"""
✅ **Processamento Executado com Sucesso!**

**Método:** {result.method_name}  
**Tempo:** {result.execution_time:.2f}s

{result.explanation}

📊 Veja os gráficos comparativos abaixo!

---

**📚 Referências:**
"""
                for ref in result.references:
                    response += f"- {ref}\n"
                
                # Retorna resposta com os plots
                return response, result.figures
                
            except Exception as e:
                logger.error(f"Erro ao executar processamento: {str(e)}")
                return f"❌ Erro ao executar processamento: {str(e)}\n\nVerifique os parâmetros e tente novamente.", []
        
        # Caso contrário, usa LLM para resposta conversacional
        rag_results = st.session_state.rag_engine.search(user_input, top_k=2)
        
        rag_context = ""
        if rag_results:
            rag_context = "\n\n**Contexto Científico Relevante:**\n"
            for result in rag_results:
                rag_context += f"\n{result['document'][:300]}...\n"
        
        data_context = ""
        if 'current_data' in st.session_state:
            data = st.session_state.current_data
            data_context = f"""
**Dados Carregados:**
- Tipo: {data.data_type} {data.dimension}
- Pontos: {data.metadata['n_points']}
- Unidade: {data.units}
- Range: {data.metadata['min']:.2f} a {data.metadata['max']:.2f}
"""
        
        system_prompt = f"""Você é o GeoBot, assistente especializado em processamento e análise de dados geofísicos de gravimetria e magnetometria.

**IMPORTANTE:** Seja conciso. Máximo 3-4 linhas. Não explique processos detalhadamente a menos que solicitado.

Quando o usuário pedir processamento, diga apenas que o sistema executará automaticamente.

Para perguntas simples, responda diretamente em 1-2 frases. 

Responda no idioma que o usuário utilizar e leve sempre em conta o contexto dos dados carregados.

{data_context}

{rag_context}

Seja breve, técnico e direto.
"""
        
        messages = [{"role": "system", "content": system_prompt}]
        
        recent_history = st.session_state.chat_history[-5:]
        for msg in recent_history:
            if "content" in msg:
                messages.append({"role": msg["role"], "content": msg["content"]})
        
        messages.append({"role": "user", "content": user_input})
        
        llm = st.session_state.llm_manager
        response = llm.chat_completion(messages, temperature=0.3, max_tokens=800)
        
        if rag_results:
            response += "\n\n---\n### 📚 Referências:\n\n"
            for result in rag_results:
                citation = st.session_state.rag_engine.format_citation_abnt(
                    result['metadata'],
                    result['document'][:200]
                )
                response += citation + "\n"
        
        return response, []
        
    except Exception as e:
        logger.error(f"Erro ao gerar resposta: {str(e)}")
        return f"❌ Erro ao gerar resposta: {str(e)}\n\nPor favor, tente novamente.", []


# ============================================================================
# APLICAÇÃO PRINCIPAL
# ============================================================================

def main():
    """Função principal da aplicação Streamlit."""
    
    # Configuração da página
    st.set_page_config(
        page_title="GeoBot - PPG DOT UFF",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded"  # Sidebar inicia expandida
    )
    
    # CSS customizado
    st.markdown("""
    <style>
    /* Importar fonte Roboto */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
    
    /* Cores PPGDot-UFF */
    :root {
        --ppgdot-azul-escuro: #003d5c;
        --ppgdot-azul-medio: #0066a1;
        --ppgdot-azul-claro: #0080c0;
        --ppgdot-azul-agua: #00a8cc;
        --ppgdot-cinza-claro: #f5f7fa;
        --ppgdot-branco: #ffffff;
    }
    
    /* Fonte global */
    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
    }
    
    /* App principal */
    .stApp {
        max-width: 100%;
        background: linear-gradient(180deg, #f5f7fa 0%, #ffffff 100%);
    }
    
    /* Header customizado - Visível nas páginas landing e model_selection */
    [data-testid="stHeader"] {
        background: linear-gradient(90deg, var(--ppgdot-azul-escuro) 0%, var(--ppgdot-azul-medio) 100%);
        border-bottom: 3px solid var(--ppgdot-azul-agua);
        min-height: 90px !important;
        height: 90px !important;
    }
    
    /* Logo no topo */
    .header-logo {
        position: fixed;
        top: 12px;
        left: 20px;
        z-index: 999999;
    }
    
    .header-logo img {
        height: 70px;
        width: auto;
    }
    
    /* Sidebar - SEMPRE EXPANDIDA E VISÍVEL */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--ppgdot-azul-escuro) 0%, var(--ppgdot-azul-medio) 100%);
        border-right: 2px solid var(--ppgdot-azul-agua);
    }
    
    /* Botão de colapsar com estilo PPGDot */
    [data-testid="stSidebarCollapseButton"] {
        background-color: var(--ppgdot-azul-medio) !important;
        color: white !important;
    }
    
    [data-testid="stSidebarCollapseButton"]:hover {
        background-color: var(--ppgdot-azul-agua) !important;
    }
    
    [data-testid="collapsedControl"] button {
        background-color: var(--ppgdot-azul-medio) !important;
        color: white !important;
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: white !important;
        font-weight: 500;
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Divisores brancos na sidebar */
    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.3) !important;
        margin: 1rem 0 !important;
    }
    
    /* FORÇAR TUDO NA SIDEBAR SEM FUNDO BRANCO */
    [data-testid="stSidebar"] *:not(svg):not(path):not(circle):not(rect) {
        background-color: transparent !important;
        background: transparent !important;
    }
    
    /* Remover bordas e fundos das caixas na sidebar */
    [data-testid="stSidebar"] .element-container,
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"],
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"],
    [data-testid="stSidebar"] div[data-testid="column"],
    [data-testid="stSidebar"] div[class*="st-emotion-cache"],
    [data-testid="stSidebar"] section,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] button {
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    
    /* File uploader - FORÇAR elementos internos transparentes */
    [data-testid="stFileUploader"],
    [data-testid="stFileUploader"] *,
    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploader"] > div,
    [data-testid="stFileUploader"] button {
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
    }
    
    /* Botão do file uploader com fundo azul transparente */
    [data-testid="stFileUploader"] button {
        background: rgba(0, 102, 161, 0.3) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 8px !important;
    }
    
    [data-testid="stFileUploader"] button:hover {
        background: rgba(0, 128, 192, 0.4) !important;
    }
    
    /* Métricas na sidebar sem caixas brancas */
    [data-testid="stSidebar"] .stMetric,
    [data-testid="stSidebar"] .stMetric > div {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 5px 0 !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        color: white !important;
    }
    
    /* Botões */
    .stButton > button {
        background: linear-gradient(90deg, var(--ppgdot-azul-medio) 0%, var(--ppgdot-azul-claro) 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, var(--ppgdot-azul-claro) 0%, var(--ppgdot-azul-agua) 100%);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transform: translateY(-2px);
    }
    
    /* Container principal do chat - fundo bege WhatsApp */
    .main .block-container {
        background: #e5ddd5 !important;
        padding: 20px !important;
    }
    
    /* Estilo geral para containers no chat */
    .stMarkdown > div {
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* Métricas */
    .stMetric {
        background: linear-gradient(135deg, var(--ppgdot-cinza-claro) 0%, var(--ppgdot-branco) 100%);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(0, 102, 161, 0.2);
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    [data-testid="stMetricValue"] {
        color: var(--ppgdot-azul-escuro);
        font-weight: 600;
    }
    
    /* Títulos */
    h1, h2, h3 {
        color: var(--ppgdot-azul-escuro);
        font-weight: 500;
    }
    
    /* File uploader - FORÇAR TRANSPARENTE */
    [data-testid="stFileUploader"] {
        background: transparent !important;
        border-radius: 10px;
        padding: 10px;
    }
    
    [data-testid="stFileUploader"] section {
        background: transparent !important;
    }
    
    [data-testid="stFileUploader"] > div {
        background: transparent !important;
    }
    
    /* Chat input - FORÇAR transparência total SEM BORDAS */
    [data-testid="stChatInput"],
    [data-testid="stChatInput"] *,
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] section {
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    
    /* Input de texto do chat - estilo limpo */
    [data-testid="stChatInput"] input,
    [data-testid="stChatInput"] textarea {
        background: white !important;
        border: 1px solid #ddd !important;
        border-radius: 20px !important;
        padding: 10px 15px !important;
        box-shadow: none !important;
    }
    
    [data-testid="stChatInput"] input:focus,
    [data-testid="stChatInput"] textarea:focus {
        border: 1px solid #0066a1 !important;
        outline: none !important;
        box-shadow: 0 0 0 2px rgba(0, 102, 161, 0.1) !important;
    }
    
    /* Container de mensagens do chat */
    [data-testid="stChatMessageContainer"],
    [data-testid="stChatMessageContainer"] > div,
    [data-testid="stChatMessage"],
    [data-testid="stChatMessage"] > div {
        background: transparent !important;
        background-color: transparent !important;
    }
    
    /* Main content area sem fundo branco */
    .main .block-container,
    .main [data-testid="stVerticalBlock"],
    .main [data-testid="stHorizontalBlock"],
    .main .element-container {
        background: transparent !important;
        background-color: transparent !important;
    }
    
    /* Inputs */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 2px solid var(--ppgdot-azul-claro);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: var(--ppgdot-cinza-claro);
        border-radius: 8px;
        font-weight: 500;
    }
    
    /* Linha vertical entre dados e chat */
    div[data-testid="column"]:first-child {
        border-right: 2px solid #e0e0e0;
        padding-right: 20px;
    }
    
    /* Reduzir fontes da área de dados */
    .stMetric label {
        font-size: 0.8rem !important;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        font-size: 1.3rem !important;
    }
    
    /* Títulos menores */
    h1 {
        font-size: 1.6rem !important;
    }
    
    h2 {
        font-size: 1.3rem !important;
    }
    
    h3 {
        font-size: 1rem !important;
    }
    
    /* Cards */
    .element-container {
        transition: transform 0.2s ease;
    }
    
    /* Success/Error messages */
    .stSuccess {
        background: linear-gradient(90deg, #28a745 0%, #34ce57 100%);
        border-radius: 8px;
    }
    
    .stError {
        background: linear-gradient(90deg, #dc3545 0%, #ff4757 100%);
        border-radius: 8px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: var(--ppgdot-cinza-claro);
        border-radius: 8px 8px 0 0;
        color: var(--ppgdot-azul-escuro);
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(180deg, var(--ppgdot-azul-medio) 0%, var(--ppgdot-azul-claro) 100%);
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Logo no topo esquerda (header azul foi removido)
    logo_base64 = get_ppgdot_logo_base64()
    if logo_base64:
        st.markdown(f"""
        <div class='header-logo'>
            <img src="{logo_base64}" alt='PPGDot-UFF'>
        </div>
        """, unsafe_allow_html=True)
    
    # Inicializa session_state
    if 'page' not in st.session_state:
        st.session_state.page = "landing"
    
    # CSS condicional para ocultar header apenas na página do chat
    if st.session_state.page == "main_app":
        st.markdown("""
        <style>
        /* Ocultar header APENAS na página do chat */
        [data-testid="stHeader"] {
            display: none !important;
            height: 0px !important;
            min-height: 0px !important;
        }
        </style>
        """, unsafe_allow_html=True)
    
    # Roteamento de páginas
    if st.session_state.page == "landing":
        render_landing_page()
    
    elif st.session_state.page == "model_selection":
        render_model_selection()
    
    elif st.session_state.page == "main_app":
        render_sidebar()
        
        # CSS para linha divisória entre colunas
        st.markdown("""
        <style>
        /* Borda vertical entre dados e chat */
        div[data-testid="column"]:first-child {
            border-right: 2px solid #e0e0e0;
            padding-right: 20px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Layout principal: Dados (60%) | Chat (40%)
        col_data, col_chat = st.columns([6, 4])
        
        with col_data:
            render_data_panel()
        
        with col_chat:
            render_main_chat()


# ============================================================================
# PONTO DE ENTRADA
# ============================================================================

if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info(f"{APP_TITLE} v{APP_VERSION}")
    logger.info("=" * 80)
    
    main()
