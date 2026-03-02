// opencc-js currently ships without stable TypeScript declarations in this setup.
import { Converter } from "opencc-js";

let converter: ((input: string) => string) | undefined;

function getConverter(): (input: string) => string {
  if (!converter) {
    converter = Converter({ from: "cn", to: "hk" });
  }
  const resolved = converter;
  if (!resolved) {
    throw new Error("Failed to initialize OpenCC converter");
  }
  return resolved;
}

export function s2hk(content: string): string {
  return getConverter()(content);
}
