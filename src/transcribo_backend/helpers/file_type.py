def is_audio_file(content_type: str) -> bool:
    """
    Check if the uploaded file is an audio file.
    """
    allowed_types = ["audio"]
    file_type = content_type.split("/")[0]
    return file_type in allowed_types


def is_video_file(content_type: str) -> bool:
    """
    Check if the uploaded file is a video file.
    """
    allowed_types = ["video"]
    file_type = content_type.split("/")[0]
    return file_type in allowed_types
