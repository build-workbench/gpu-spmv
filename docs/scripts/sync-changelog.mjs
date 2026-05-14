#!/usr/bin/env node
import { readFileSync, writeFileSync, existsSync } from "fs";
import { dirname, join } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const docsDir = join(__dirname, "..");
const rootDir = join(docsDir, "..");

const sourcePath = join(rootDir, "CHANGELOG.md");
const enTargetPath = join(docsDir, "en/changelog.md");
const zhTargetPath = join(docsDir, "zh/changelog.md");

if (!existsSync(sourcePath)) {
  console.log("CHANGELOG.md not found, skipping sync");
  process.exit(0);
}

const EN_HEADER = `# Changelog

All notable changes to GPU SpMV are documented here.

`;
const ZH_HEADER = `# 更新日志

GPU SpMV 的所有重要变更都记录在此文件中。

`;

let content = readFileSync(sourcePath, "utf-8");
content = content.replace(/<!--[\s\S]*?-->\n*/g, "");

writeFileSync(enTargetPath, EN_HEADER + content);
writeFileSync(zhTargetPath, ZH_HEADER + content);

console.log(`Synced changelog to ${enTargetPath}`);
console.log(`Synced changelog to ${zhTargetPath}`);
