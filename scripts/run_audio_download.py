"""Stream yt-dlp output and distinguish missing videos from operational failures."""
import re
import subprocess
import sys


def classify_failure(message):
    # Restrictions and network failures must never be recorded as missing media.
    if re.search(r"(?i)confirm.*bot|sign in|captcha|rate.?limit|too many requests|HTTP Error (?:403|429)|not available in your country", message):
        return 21
    if re.search(r"(?i)private video|this video is unavailable|video unavailable|video has been removed|video has been deleted", message):
        return 20
    return 1


def main():
    errors = []
    with subprocess.Popen(
        [sys.executable, '-m', 'yt_dlp', *sys.argv[1:]],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding='utf-8', errors='replace',
    ) as process:
        for line in process.stdout:
            print(line, end='', flush=True)
            if 'ERROR:' in line:
                errors.append(line.strip())
        code = process.wait()
    return 0 if code == 0 else classify_failure('\n'.join(errors))


if __name__ == '__main__':
    raise SystemExit(main())
