import unittest
from scripts.run_audio_download import classify_failure


class DownloadErrorTests(unittest.TestCase):
    def test_missing_videos_are_skippable(self):
        for error in ('ERROR: Private video', 'ERROR: This video is unavailable',
                      'ERROR: Video has been removed by the uploader'):
            self.assertEqual(classify_failure(error), 20)

    def test_operational_errors_are_not_missing_media(self):
        for error in ('ERROR: ffmpeg is not installed', 'ERROR: timed out',
                      'ERROR: Requested format is not available'):
            self.assertEqual(classify_failure(error), 1)

    def test_access_restrictions_take_precedence(self):
        for error in ('ERROR: Video unavailable. Sign in to confirm you are not a bot',
                      'ERROR: HTTP Error 429: Too Many Requests',
                      'ERROR: Video unavailable. Not available in your country'):
            self.assertEqual(classify_failure(error), 21)
