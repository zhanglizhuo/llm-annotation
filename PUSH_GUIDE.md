# Push Guide (Stable SSH over 443)

This repository uses GitHub push over SSH on port 443 for better stability in restricted networks.

## 1) One-time SSH setup

Create or reuse an SSH key:

```bash
mkdir -p ~/.ssh
ssh-keygen -t ed25519 -C "school_Agri-github" -f ~/.ssh/id_ed25519_github -N ""
```

Add GitHub host config:

```bash
cat >> ~/.ssh/config <<'EOF'
Host github.com
  HostName ssh.github.com
  Port 443
  User git
  IdentityFile ~/.ssh/id_ed25519_github
  IdentitiesOnly yes
EOF
chmod 600 ~/.ssh/config
```

Show public key and add it to GitHub:

```bash
cat ~/.ssh/id_ed25519_github.pub
```

GitHub path:
- Settings
- SSH and GPG keys
- New SSH key

## 2) Repository remote setup

Use SSH remote URL:

```bash
git remote set-url origin git@github.com:zhanglizhuo/llm-annotation.git
git remote -v
```

## 3) Verify and push

Verify SSH auth:

```bash
ssh -T git@github.com
```

Expected output includes:
- "Hi <username>! You've successfully authenticated..."

Push branch:

```bash
git push origin main
```

## 4) What to keep out of git

Keep large runtime artifacts out of version control:
- `Annotation/logs/`
- `Annotation/results/` raw/intermediate files
- model caches (for example `~/.cache/huggingface/`)

Keep lightweight deliverables:
- code and scripts
- docs and metadata
- summary JSON/CSV used for reporting

## 5) Troubleshooting

`Permission denied (publickey)`:
- Confirm the correct public key was added to GitHub account.
- Confirm `~/.ssh/config` points `github.com` to `id_ed25519_github`.
- Re-test with `ssh -T git@github.com`.

`Connection timed out` on HTTPS:
- Prefer SSH over 443 as configured above.

Need to move commit through another machine:

```bash
# source machine
git bundle create llm-annotation-main.bundle main

# destination machine
git clone https://github.com/zhanglizhuo/llm-annotation.git
cd llm-annotation
git pull /path/to/llm-annotation-main.bundle main
git push origin main
```
