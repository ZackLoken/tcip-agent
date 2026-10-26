/**
 * The frontend's non-test sources, their reads of the generated route module, their backend path
 * literals and their declarations, as the TypeScript compiler parses them.
 *
 * A non-test source is a `.ts` or `.tsx` file under the source directory whose name carries no
 * `.test.` and that lies outside the directory's `test` folder. A read is a property access whose
 * object is the identifier `ROUTES`; a destructuring or an element access of `ROUTES` is not one.
 * A backend path literal is a string or template literal starting `/api/` or `/ws/`.
 *
 * Run as `node scripts/routeReads.mjs <src>`, it prints one JSON object over the non-test sources
 * under `<src>`: `sources`, their sorted absolute paths; `reads`, the sorted names they read;
 * `literals`, each backend path literal as its `file` and `text`; `shapes`, each declared object
 * shape; `unions`, each string literal union a type alias or a property declares; and
 * `constants`, each exported string constant.
 */
import { readdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { pathToFileURL } from "node:url";

import ts from "typescript";

/**
 * Return whether `relative`, a path relative to the source directory, is a non-test source.
 *
 * @param {string} relative
 */
export function isNonTestSource(relative) {
  const parts = relative.split(/[\\/]/);
  return /\.tsx?$/.test(relative) && !parts.at(-1).includes(".test.") && parts[0] !== "test";
}

/**
 * Return the non-test sources under the source directory `src`, as sorted absolute paths.
 *
 * @param {string} src
 * @returns {string[]}
 */
export function nonTestSources(src) {
  return readdirSync(src, { recursive: true })
    .filter(isNonTestSource)
    .map((file) => join(src, file))
    .sort();
}

/** Return the source file `file` parses to on its own. */
function parsed(file) {
  return ts.createSourceFile(file, readFileSync(file, "utf8"), ts.ScriptTarget.Latest);
}

/** Call `see` on every syntax node of each of `files`, as `parse` reads it, with its file. */
function eachNode(files, see, parse = parsed) {
  const visit = (file) => (node) => {
    see(node, file);
    ts.forEachChild(node, visit(file));
  };
  for (const file of files) visit(file)(parse(file));
}

/** Return the sorted names `files` read as `ROUTES.<name>`. */
export function routeReads(files) {
  const names = new Set();
  eachNode(files, (node) => {
    if (
      ts.isPropertyAccessExpression(node) &&
      ts.isIdentifier(node.expression) &&
      node.expression.text === "ROUTES"
    ) {
      names.add(node.name.text);
    }
  });
  return [...names].sort();
}

/** Return each backend path literal in `files` as its `file` and its `text`. */
export function backendPathLiterals(files) {
  const found = [];
  eachNode(files, (node, file) => {
    const text =
      ts.isStringLiteral(node) || ts.isNoSubstitutionTemplateLiteral(node)
        ? node.text
        : ts.isTemplateExpression(node)
          ? node.head.text
          : null;
    if (text !== null && /^\/(api|ws)\//.test(text)) found.push({ file, text });
  });
  return found;
}

/**
 * Return each object shape `files` declare, an interface or a type alias of an object literal
 * type, as its `file`, its `name` and the sorted names of all its property `fields`, inherited
 * ones among them, as the type checker resolves them under the `tsconfig.json` nearest the first
 * file.
 */
export function declaredShapes(files) {
  const config = ts.findConfigFile(dirname(files[0]), ts.sys.fileExists);
  const { options } = ts.parseJsonConfigFileContent(
    ts.readConfigFile(config, ts.sys.readFile).config,
    ts.sys,
    dirname(config),
  );
  const program = ts.createProgram(files, options);
  const checker = program.getTypeChecker();
  const found = [];
  eachNode(
    files,
    (node, file) => {
      const shape =
        ts.isInterfaceDeclaration(node) ||
        (ts.isTypeAliasDeclaration(node) && ts.isTypeLiteralNode(node.type));
      if (!shape) return;
      const fields = checker
        .getPropertiesOfType(checker.getTypeAtLocation(node.name))
        .map((property) => property.name);
      found.push({ file, name: node.name.text, fields: fields.sort() });
    },
    (file) => program.getSourceFile(file),
  );
  return found;
}

/** Return the sorted string members of `type` when it is a union of string literals and `null`. */
function stringMembers(type) {
  if (type === undefined || !ts.isUnionTypeNode(type)) return null;
  const members = [];
  for (const member of type.types) {
    if (!ts.isLiteralTypeNode(member)) return null;
    if (ts.isStringLiteral(member.literal)) members.push(member.literal.text);
    else if (member.literal.kind !== ts.SyntaxKind.NullKeyword) return null;
  }
  return members.length > 0 ? members.sort() : null;
}

/**
 * Return each union of string literal types `files` declare as a type alias's or a property's
 * own type, `null` members left out, as its `file`, its `alias` (the type alias it is the whole
 * of, `null` for a property) and its sorted string `members`.
 */
export function stringUnions(files) {
  const found = [];
  eachNode(files, (node, file) => {
    const alias = ts.isTypeAliasDeclaration(node);
    if (!alias && !ts.isPropertySignature(node)) return;
    const members = stringMembers(node.type);
    if (members !== null) found.push({ file, alias: alias ? node.name.text : null, members });
  });
  return found;
}

/** Return each exported `const` of `files` a string literal initializes, as `file`, `name`, `value`. */
export function stringConstants(files) {
  const found = [];
  eachNode(files, (node, file) => {
    if (
      !ts.isVariableStatement(node) ||
      !(node.declarationList.flags & ts.NodeFlags.Const) ||
      !node.modifiers?.some((modifier) => modifier.kind === ts.SyntaxKind.ExportKeyword)
    ) {
      return;
    }
    for (const declaration of node.declarationList.declarations) {
      if (declaration.initializer && ts.isStringLiteral(declaration.initializer)) {
        found.push({ file, name: declaration.name.text, value: declaration.initializer.text });
      }
    }
  });
  return found;
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  const sources = nonTestSources(process.argv[2]);
  process.stdout.write(
    JSON.stringify({
      sources,
      reads: routeReads(sources),
      literals: backendPathLiterals(sources),
      shapes: declaredShapes(sources),
      unions: stringUnions(sources),
      constants: stringConstants(sources),
    }),
  );
}
