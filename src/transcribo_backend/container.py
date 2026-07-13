from dcc_backend_common.usage_tracking import UsageTrackingService
from dependency_injector import containers, providers

from transcribo_backend.agents.speaker_inference_agent import SpeakerInferenceAgent
from transcribo_backend.agents.summarize_agent import SummarizeAgent
from transcribo_backend.agents.transcript_cleanup_agent import TranscriptCleanupAgent
from transcribo_backend.services.summarization_service import SummarizationService
from transcribo_backend.services.transcript_postprocessing_service import TranscriptPostProcessingService
from transcribo_backend.services.whisper_service import WhisperService
from transcribo_backend.utils.app_config import AppConfig


class Container(containers.DeclarativeContainer):
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

    speaker_inference_agent: providers.Singleton[SpeakerInferenceAgent] = providers.Singleton(
        SpeakerInferenceAgent,
        config=app_config,
    )

    transcript_cleanup_agent: providers.Singleton[TranscriptCleanupAgent] = providers.Singleton(
        TranscriptCleanupAgent,
        config=app_config,
    )

    transcript_postprocessing_service: providers.Singleton[TranscriptPostProcessingService] = providers.Singleton(
        TranscriptPostProcessingService,
        app_config=app_config,
        speaker_inference_agent=speaker_inference_agent,
        transcript_cleanup_agent=transcript_cleanup_agent,
    )
