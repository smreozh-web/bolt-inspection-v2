import os
import re
import runpy
import shutil
import subprocess
import threading
import time
import atexit
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PORT = 5001

# 기존 비밀번호 설정은 변경하지 않습니다.
password = os.environ.get('SCREW_WEB_PASSWORD', '012345678910')
if len(password) < 12:
    raise SystemExit('비밀번호는 12자 이상으로 설정하세요.')
os.environ['SCREW_WEB_PASSWORD'] = password

print('====================================', flush=True)
print(' AI Screw Inspection V4 보안 서버', flush=True)
print('====================================', flush=True)
print(f'로컬 주소: http://127.0.0.1:{PORT}', flush=True)

exe = next((str(p) for p in (ROOT / 'cloudflared.exe', ROOT / 'cloudflared') if p.is_file()), None)
exe = exe or shutil.which('cloudflared')
tunnel = None
stop_event = threading.Event()
registered = threading.Event()
# 중요: 점(.) 앞에는 역슬래시 하나만 사용합니다.
url_pattern = re.compile(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com')
seen_urls = set()


def dns_ready(host):
    """Cloudflare 공개 DNS로 조회하고, 응답의 Address 행을 확인합니다."""
    try:
        result = subprocess.run(
            ['nslookup', host, '1.1.1.1'],
            capture_output=True, text=True, errors='replace', timeout=8,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
        )
        output = result.stdout + '\n' + result.stderr
        # 로컬 Windows 언어와 무관하게 조회 결과의 IP 형식 확인
        if result.returncode != 0 or re.search(r'NXDOMAIN|Non-existent domain|찾을 수 없', output, re.I):
            return False
        # nslookup 결과에서 질의 대상 이름이 출력되고 IP가 하나 이상 있어야 함
        return host.lower() in output.lower() and bool(
            re.search(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', output.split(host, 1)[-1])
            or re.search(r'\b[0-9a-f]{0,4}:[0-9a-f:]+\b', output.split(host, 1)[-1], re.I)
        )
    except (OSError, subprocess.TimeoutExpired):
        return False


def check_dns(url):
    host = url.removeprefix('https://')
    deadline = time.monotonic() + 120
    while not stop_event.is_set() and time.monotonic() < deadline:
        if tunnel is None or tunnel.poll() is not None:
            print('[Cloudflare] 터널이 종료되어 DNS 확인을 중단합니다.', flush=True)
            return
        if registered.is_set() and dns_ready(host):
            print('[Cloudflare] DNS 확인 성공. 외부 접속을 시도할 수 있습니다.', flush=True)
            return
        stop_event.wait(5)
    if not stop_event.is_set():
        print('[Cloudflare] DNS 확인이 120초 동안 완료되지 않았습니다.', flush=True)
        print('[Cloudflare] 위 링크가 접속되지 않으면 터널을 재실행해 보세요.', flush=True)


def watch_tunnel():
    if tunnel is None or tunnel.stdout is None:
        return
    for line in tunnel.stdout:
        message = line.strip()
        if 'Registered tunnel connection' in message:
            registered.set()
            print('[Cloudflare] 터널 연결 등록 완료', flush=True)

        match = url_pattern.search(message)
        if match:
            url = match.group(0)
            if url not in seen_urls:
                seen_urls.add(url)
                print('\n' + '=' * 60, flush=True)
                print('외부 접속 링크: ' + url, flush=True)
                print('이 주소는 실행할 때마다 변경될 수 있습니다.', flush=True)
                print('DNS 확인 중... (확인 전에는 접속이 안 될 수 있습니다.)', flush=True)
                print('=' * 60 + '\n', flush=True)
                threading.Thread(target=check_dns, args=(url,), daemon=True).start()

        if any(word in message.lower() for word in ('error', 'failed', 'unable', ' err ')):
            print('[Cloudflare] ' + message, flush=True)
    if not stop_event.is_set():
        print('[Cloudflare] 터널 로그가 종료되었습니다.', flush=True)


def stop_tunnel():
    stop_event.set()
    if tunnel is not None and tunnel.poll() is None:
        tunnel.terminate()
        try:
            tunnel.wait(timeout=5)
        except subprocess.TimeoutExpired:
            tunnel.kill()
            tunnel.wait()


atexit.register(stop_tunnel)

if exe:
    try:
        tunnel = subprocess.Popen(
            [exe, 'tunnel', '--url', f'http://127.0.0.1:{PORT}',
             '--no-autoupdate', '--loglevel', 'info'],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', errors='replace', bufsize=1,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
        )
        threading.Thread(target=watch_tunnel, daemon=True).start()
        print('외부 접속 링크 생성 중... (잠시 기다려 주세요)', flush=True)
    except OSError as exc:
        print('Cloudflare Tunnel 실행 실패:', exc, flush=True)
else:
    print('cloudflared.exe를 프로젝트 폴더에 넣거나 cloudflared를 설치하세요.', flush=True)

try:
    runpy.run_path(str(ROOT / 'app.py'), run_name='__main__')
finally:
    stop_tunnel()
