from django.urls import path

from . import views

app_name = "transcribe"

urlpatterns = [
    path("upload/", views.transcribe_upload, name="transcribe-upload"),
    path("live/", views.transcribe_live_chunk, name="transcribe-live"),
]
