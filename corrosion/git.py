import subprocess


def get_current_branch():
    result = subprocess.run(
        ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode == 0:
        return result.stdout.strip()
    else:
        raise Exception(f'Error getting current branch: {result.stderr}')


def get_commit_id():
    result = subprocess.run(
        ['git', 'rev-parse', 'HEAD'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode == 0:
        return result.stdout.strip()
    else:
        raise Exception(f'Error getting commit ID: {result.stderr}')
