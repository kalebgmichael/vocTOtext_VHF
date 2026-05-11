from django.urls import path

from transcribe.consumers import LiveTranscribeConsumer
from transcribe.consumers import PlaybackConsumer
from transcribe.consumers import RadioTranscribeConsumer

websocket_urlpatterns = [
    path("ws/transcribe/live/", LiveTranscribeConsumer.as_asgi()),
    path("ws/transcribe/radio/", RadioTranscribeConsumer.as_asgi()),
    path("ws/transcribe/playback/", PlaybackConsumer.as_asgi()),
]
