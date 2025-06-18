/**
 * @fileoverview Angular service wrapper for the 3P package highlightjs.
 * @see google3/third_party/javascript/highlightjs/README.md
 */

import 'google3/third_party/javascript/highlightjs/highlightjs_bash_raw';
import 'google3/third_party/javascript/highlightjs/highlightjs_c_raw';
import 'google3/third_party/javascript/highlightjs/highlightjs_cpp_raw';
import 'google3/third_party/javascript/highlightjs/highlightjs_css_raw';
import 'google3/third_party/javascript/highlightjs/highlightjs_go_raw';
import 'google3/third_party/javascript/highlightjs/highlightjs_html_raw';
import 'google3/third_party/javascript/highlightjs/highlightjs_java_raw';
import 'google3/third_party/javascript/highlightjs/highlightjs_kotlin_raw';
import 'google3/third_party/javascript/highlightjs/highlightjs_python_raw';
import 'google3/third_party/javascript/highlightjs/highlightjs_sql_raw';
import 'google3/third_party/javascript/highlightjs/highlightjs_typescript_raw';

import {Injectable} from '@angular/core';
import * as hljs from 'google3/third_party/javascript/highlightjs/highlightjs_raw_raw';

hljs.registerLanguage('python', hljs_python);
hljs.registerLanguage('java', hljs_java);
hljs.registerLanguage('go', hljs_go);
hljs.registerLanguage('typescript', hljs_typescript);
hljs.registerLanguage('javascript', hljs_javascript);
hljs.registerLanguage('c', hljs_c);
hljs.registerLanguage('cpp', hljs_cpp);
hljs.registerLanguage('kotlin', hljs_kotlin);
hljs.registerLanguage('css', hljs_css);
hljs.registerLanguage('bash', hljs_bash);
hljs.registerLanguage('html', hljs_html);
hljs.registerLanguage('sql', hljs_sql);

/** A service for syntax highlighting. */
@Injectable({providedIn: 'root'})
export class SyntaxHighlightService {
  highlight(code: string, fileName?: string) {
    if (!fileName) {
      return hljs.highlightAuto(code);
    }
    const language = guessLanguage(fileName);
    if (!language) {
      return hljs.highlightAuto(code);
    }
    return hljs.highlight(language, code);
  }
}

function guessLanguage(fileName: string): string|undefined {
  if (fileName.endsWith('.py')) {
    return 'python';
  } else if (fileName.endsWith('.java')) {
    return 'java';
  } else if (fileName.endsWith('.go')) {
    return 'go';
  } else if (fileName.endsWith('.ts')) {
    return 'typescript';
  } else if (fileName.endsWith('.js')) {
    return 'javascript';
  } else if (fileName.endsWith('.c')) {
    return 'c';
  } else if (fileName.endsWith('.cc') || fileName.endsWith('.cpp')) {
    return 'cpp';
  } else if (fileName.endsWith('.kt')) {
    return 'kotlin';
  } else if (fileName.endsWith('.css')) {
    return 'css';
  } else if (fileName.endsWith('.sh')) {
    return 'bash';
  } else if (fileName.endsWith('.html')) {
    return 'html';
  } else if (fileName.endsWith('.sql')) {
    return 'sql';
  }
  return undefined;
}
