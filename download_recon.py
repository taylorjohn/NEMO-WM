import urllib.request, os, time

url = 'http://rail.eecs.berkeley.edu/datasets/recon-navigation/recon_dataset.tar.gz'
out = './recon_dataset.tar.gz'

for attempt in range(10):
    try:
        existing = os.path.getsize(out) if os.path.exists(out) else 0
        req = urllib.request.Request(url)
        if existing > 0:
            req.add_header('Range', f'bytes={existing}-')
            print(f'Resuming from {existing/1e9:.2f}GB...')
        with urllib.request.urlopen(req) as r, open(out, 'ab') as f:
            total = int(r.headers.get('Content-Length', 0)) + existing
            downloaded = existing
            while chunk := r.read(1024*1024):
                f.write(chunk)
                downloaded += len(chunk)
                print(f'{downloaded/1e9:.2f}/{total/1e9:.1f}GB', end=chr(13))
        print('Done!')
        break
    except Exception as e:
        print(f'Attempt {attempt+1} failed: {e}. Retrying in 10s...')
        time.sleep(10)
