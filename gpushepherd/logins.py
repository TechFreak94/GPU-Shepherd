"""Logins of apps installed from the App Center: who can log in, and setting a new password.

Each app keeps its accounts its own way:
    Hermes Agent,   one user name (Open Notebook: only a password) and password in the instance's
    SillyTavern,    environment, read when it starts
    Open Notebook
    AnythingLLM     one password (AUTH_TOKEN) in its own settings file, server.env; in multi-user
                    mode, accounts in its database (storage/anythingllm.db)
    Open WebUI      accounts in its database (data/webui.db)
    LibreChat       accounts in its MongoDB database
    Langflow        accounts in its database (data/langflow.db); the admin is made at its first
                    start from the instance's environment
    LLM server      no accounts: llama-swap takes its API key as the password, with any user name,
                    when a browser opens it (the key is in its model catalogue)
A password GPU Shepherd can read (the environment, server.env) is shown. A database only keeps a
hash of each password, so for those the accounts are listed and a new password can be set, by a
short script run inside the app's running container with the app's own libraries. The new
password reaches the script in its environment, never on a command line.

This module is pure: it builds the scripts and reads their output; the service runs them.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from . import instance_templates as tpl
from .config import ConfigError, Instance

MIN_PASSWORD = 8
SINGLE_USER = "(password only)"        # AnythingLLM without multi-user mode: no user name
ANY_USER = "(any user name)"           # the LLM server: only the password, its API key, is checked
# apps whose login is in the instance's environment (Template.login), read when they start
ENV_LOGINS = ("hermes", "sillytavern", "open-notebook")


@dataclass
class Account:
    user: str
    role: str = ""
    password: str = ""                 # only where GPU Shepherd can read it


@dataclass
class Exec:
    """A script to run inside the app's container: `cmd`, with `env` and in `workdir`."""
    cmd: list[str]
    env: dict[str, str] = field(default_factory=dict)
    workdir: str = ""


@dataclass
class Method:
    app: str
    where: str                         # where the app keeps its logins, in a few words
    in_container: bool                 # reading or changing them runs a script in the running app
    note: str = ""


def method(inst: Instance) -> Method | None:
    """How an instance's logins are read and reset (None: GPU Shepherd doesn't know its logins)."""
    if inst.kind == "llama-swap":
        return Method("llm-server", "its API key", False,
                      "When your browser asks for a user name and password, type any user name and this API key "
                      "as the password. Apps connected to it already have the key. To change it, use New API key "
                      "in its Models page, Server settings; apps connected to it then need the new key too.")
    app = next((t.key for t in tpl.TEMPLATES if tpl.matches(t, inst)), "")    # older ones by their image
    if app in ENV_LOGINS:
        t = tpl.get(app)
        return Method(app, "its environment" + (" (the dashboard login)" if app == "hermes" else ""), False,
                      f"Setting a new password restarts {t.title} if it is running.")
    if app == "anythingllm":
        return Method("anythingllm", "its settings file, server.env, or in multi-user mode its database", True,
                      "Without a password anyone who can open AnythingLLM can use it. Setting its password "
                      "restarts AnythingLLM; in multi-user mode accounts change at once.")
    if app == "open-webui":
        return Method("open-webui", "its database", True,
                      "The first account made in Open WebUI is its admin.")
    if app == "librechat":
        return Method("librechat", "its MongoDB database", True,
                      "The first account made in LibreChat is its admin. Its database starts with it.")
    if app == "langflow":
        return Method("langflow", "its database", True,
                      "The admin was made from LANGFLOW_SUPERUSER and LANGFLOW_SUPERUSER_PASSWORD when "
                      "Langflow first started; changing those later does nothing, so set a new password here.")
    return None


# --------------------------------------------------------------------------- what GPU Shepherd can read

def readable(m: Method, inst: Instance, settings_file: str | None, api_key: str = "") -> list[Account]:
    """The accounts of a method that keeps them outside a database."""
    if m.app == "llm-server":
        return [Account(ANY_USER, "API key", api_key)] if api_key else []
    if m.app in ENV_LOGINS:
        user_var, password_var = tpl.get(m.app).login
        user = inst.env.get(user_var, "") if user_var else SINGLE_USER
        if not user or not inst.env.get(password_var):
            return []
        return [Account(user, "dashboard" if m.app == "hermes" else "login", inst.env[password_var])]
    if m.app == "anythingllm":
        token = env_value(settings_file or "", "AUTH_TOKEN")
        return [Account(SINGLE_USER, "everyone", token)] if token else []
    return []


def env_value(text: str, key: str) -> str:
    """A value from a .env-style file (KEY='value', KEY="value" or KEY=value)."""
    for line in text.splitlines():
        mt = re.match(rf"^\s*{re.escape(key)}\s*=\s*(.*?)\s*$", line)
        if mt:
            return mt.group(1).strip("'\"")
    return ""


def set_env_value(text: str, key: str, value: str) -> str:
    """The .env-style file with KEY set to value, added at the end if it wasn't there."""
    line = f"{key}='{value}'"
    lines = text.splitlines()
    for i, old in enumerate(lines):
        if re.match(rf"^\s*{re.escape(key)}\s*=", old):
            lines[i] = line
            break
    else:
        lines.append(line)
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- scripts run in the app

_LANGFLOW = """
import json, os, sqlite3
db = sqlite3.connect("/app/data/langflow.db")
if os.environ.get("GS_USER"):
    from langflow.services.auth.utils import get_password_hash
    n = db.execute("update user set password=? where username=?",
                   (get_password_hash(os.environ["GS_PASSWORD"]), os.environ["GS_USER"])).rowcount
    db.commit()
    print(json.dumps({"updated": n}))
else:
    print(json.dumps([{"user": u, "role": "admin" if s else "user"}
                      for u, s in db.execute("select username, is_superuser from user order by username")]))
"""

_OPEN_WEBUI = """
import json, os, sqlite3
db = sqlite3.connect(os.path.join(os.environ.get("DATA_DIR", "/app/backend/data"), "webui.db"))
if os.environ.get("GS_USER"):
    import bcrypt
    pw = bcrypt.hashpw(os.environ["GS_PASSWORD"].encode()[:72], bcrypt.gensalt()).decode()
    n = db.execute("update auth set password=? where lower(email)=lower(?)", (pw, os.environ["GS_USER"])).rowcount
    db.commit()
    print(json.dumps({"updated": n}))
else:
    print(json.dumps([{"user": e, "role": r} for e, r in
                      db.execute("select email, role from user order by role != 'admin', created_at")]))
"""

# run from LibreChat's config folder, like its own `npm run reset-password`
_LIBRECHAT = """
const path = require('path');
require('module-alias')({ base: path.resolve('/app/api') });
const mongoose = require('mongoose');
const bcrypt = require('bcryptjs');
const { User } = require('@librechat/data-schemas').createModels(mongoose);
require('./helpers');
const { connectDb } = require('~/db/connect');
(async () => {
  await connectDb();
  if (process.env.GS_USER) {
    const r = await User.updateOne({ email: process.env.GS_USER },
      { password: bcrypt.hashSync(process.env.GS_PASSWORD, 10), passwordVersion: Date.now() });
    console.log(JSON.stringify({ updated: r.matchedCount }));
  } else {
    const users = await User.find({}, 'email role provider').sort({ createdAt: 1 });
    console.log(JSON.stringify(users.map(u => ({ user: u.email, role: (u.role || '').toLowerCase() }))));
  }
  process.exit(0);
})().catch(e => { console.error(e.message || e); process.exit(1); });
"""

_ANYTHINGLLM = """
const bcrypt = require('bcryptjs');
const { PrismaClient } = require('@prisma/client');
const prisma = new PrismaClient();
(async () => {
  const multi = await prisma.system_settings.findFirst({ where: { label: 'multi_user_mode' } });
  if (process.env.GS_USER) {
    const r = await prisma.users.updateMany({ where: { username: process.env.GS_USER },
      data: { password: bcrypt.hashSync(process.env.GS_PASSWORD, 10) } });
    console.log(JSON.stringify({ updated: r.count }));
  } else {
    const users = await prisma.users.findMany({ select: { username: true, role: true }, orderBy: { id: 'asc' } });
    console.log(JSON.stringify({ multi_user: multi?.value === 'true',
                                 users: users.map(u => ({ user: u.username, role: u.role })) }));
  }
  process.exit(0);
})().catch(e => { console.error(e.message || e); process.exit(1); });
"""


def script(m: Method, user: str = "", password: str = "") -> Exec:
    """The script that lists a method's accounts, or with `user` sets that account's password."""
    env = {"GS_USER": user, "GS_PASSWORD": password} if user else {}
    if m.app == "langflow":
        return Exec(["python", "-c", _LANGFLOW], env)
    if m.app == "open-webui":
        return Exec(["python3", "-c", _OPEN_WEBUI], env)
    if m.app == "librechat":
        return Exec(["node", "-e", _LIBRECHAT], env, "/app/config")
    if m.app == "anythingllm":
        return Exec(["node", "-e", _ANYTHINGLLM], env, "/app/server")
    raise ConfigError(f"{m.app} keeps no accounts in a database")


def parse_accounts(output: str) -> tuple[list[Account], bool]:
    """The accounts a listing script printed (its last line is JSON), and whether the app keeps
    accounts at all (AnythingLLM only does in multi-user mode)."""
    data = _last_json(output)
    multi = True
    if isinstance(data, dict) and "users" in data:
        data, multi = data["users"], bool(data.get("multi_user"))
    if not isinstance(data, list):
        raise ConfigError("couldn't read the app's accounts: " + last_line(output))
    return [Account(str(a.get("user") or ""), str(a.get("role") or "")) for a in data if a.get("user")], multi


def parse_updated(output: str) -> int:
    data = _last_json(output)
    if not isinstance(data, dict) or "updated" not in data:
        raise ConfigError("couldn't set the password: " + last_line(output))
    return int(data["updated"])


def _last_json(output: str):
    for line in reversed(output.strip().splitlines()):
        line = line.strip()
        if line.startswith(("[", "{")):
            try:
                return json.loads(line)
            except ValueError:
                return None
    return None


def last_line(output: str) -> str:
    lines = [x for x in output.strip().splitlines() if x.strip()]
    return lines[-1][:300] if lines else "no output"


def check_password(password: str) -> None:
    if len(password) < MIN_PASSWORD:
        raise ConfigError(f"Password: use at least {MIN_PASSWORD} characters")
    if "'" in password or "\n" in password:
        raise ConfigError("Password: leave out quotes (') and line breaks")
