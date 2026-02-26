import { mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { spawn } from "node:child_process";

export const PATH_TRAVERSAL_ERROR = "Path traversal not allowed";
export const PATH_OUTSIDE_ROOT_ERROR = "Path outside root";

export class SandboxFS {
  constructor(public readonly rootDir: string) {}

  resolve(userPath: string): string {
    const cleanedPath = userPath.trim();
    if (!cleanedPath) {
      throw new Error("Empty path");
    }
    if (cleanedPath.startsWith("~")) {
      throw new Error(PATH_TRAVERSAL_ERROR);
    }

    const virtualPath = cleanedPath.startsWith("/")
      ? cleanedPath
      : `/${cleanedPath}`;
    if (virtualPath.includes("..")) {
      throw new Error(PATH_TRAVERSAL_ERROR);
    }

    const resolvedRoot = resolve(this.rootDir);
    const resolvedPath = resolve(resolvedRoot, virtualPath.slice(1));

    if (!resolvedPath.startsWith(resolvedRoot)) {
      throw new Error(PATH_OUTSIDE_ROOT_ERROR);
    }

    return resolvedPath;
  }
}

async function fileExists(path: string): Promise<boolean> {
  try {
    await readFile(path);
    return true;
  } catch {
    return false;
  }
}

export function makeFsTools({ rootDir }: { rootDir: string }) {
  const fs = new SandboxFS(resolve(rootDir));

  async function resolveExistingFile(path: string): Promise<string> {
    const filePath = fs.resolve(path);
    if (!(await fileExists(filePath))) {
      throw new Error(`File not found: ${path}`);
    }
    return filePath;
  }

  async function fsReadText(path: string): Promise<string> {
    const filePath = await resolveExistingFile(path);
    return readFile(filePath, "utf-8");
  }

  async function fsWriteText(path: string, text: string): Promise<string> {
    const filePath = fs.resolve(path);
    await mkdir(resolve(filePath, ".."), { recursive: true });
    await writeFile(filePath, text, "utf-8");
    return `Wrote ${path}`;
  }

  async function fsEditWithEd(path: string, script: string): Promise<string> {
    const filePath = await resolveExistingFile(path);

    await new Promise<void>((resolvePromise, rejectPromise) => {
      const child = spawn("ed", ["-s", filePath], {
        stdio: ["pipe", "pipe", "pipe"],
      });
      let stderr = "";
      let stdout = "";

      child.stderr.on("data", (chunk) => {
        stderr += String(chunk);
      });
      child.stdout.on("data", (chunk) => {
        stdout += String(chunk);
      });

      child.on("error", (error) => {
        rejectPromise(error);
      });

      child.on("close", (code) => {
        if (code !== 0) {
          const err = (stderr || stdout || "").trim();
          rejectPromise(new Error(`ed failed (code=${code}): ${err}`));
          return;
        }
        resolvePromise();
      });

      child.stdin.write(script);
      child.stdin.end();
    });

    return `Edited ${path}`;
  }

  return [fsReadText, fsWriteText, fsEditWithEd] as const;
}
