"""Dependency injection container wiring config, agents, and services."""

from dcc_backend_common.usage_tracking import UsageTrackingService
from dependency_injector import containers, providers

from transcribo_backend.agents.summarize_agent import SummarizeAgent
from transcribo_backend.agents.transcript_postprocessing_agent import TranscriptPostProcessingAgent
from transcribo_backend.services.summarization_service import SummarizationService
from transcribo_backend.services.transcript_postprocessing_service import TranscriptPostProcessingService
from transcribo_backend.services.whisper_service import WhisperService
from transcribo_backend.utils.app_config import AppConfig


class Container(containers.DeclarativeContainer):
    """Providers for the app's config, agents, and services.

    ``AppConfig.from_env()`` runs at import time of this module, so importing it
    requires a populated environment.
    """

    app_config: providers.Object[AppConfig] = providers.Object(AppConfig.from_env())

    usage_tracking_service: providers.Singleton[UsageTrackingService] = providers.Singleton(
        UsageTrackingService,
        hmac_secret=app_config.provided.hmac_secret,
    )

    whisper_service: providers.Singleton[WhisperService] = providers.Singleton(
        WhisperService,
        app_config=app_config,
    )

    summarize_agent: providers.Singleton[SummarizeAgent] = providers.Singleton(
        SummarizeAgent,
        config=app_config,
    )

    summarization_service: providers.Singleton[SummarizationService] = providers.Singleton(
        SummarizationService,
        app_config=app_config,
        summarize_agent=summarize_agent,
    )

    transcript_postprocessing_agent: providers.Singleton[TranscriptPostProcessingAgent] = providers.Singleton(
        TranscriptPostProcessingAgent,
        config=app_config,
    )

    transcript_postprocessing_service: providers.Singleton[TranscriptPostProcessingService] = providers.Singleton(
        TranscriptPostProcessingService,
        transcript_postprocessing_agent=transcript_postprocessing_agent,
    )
