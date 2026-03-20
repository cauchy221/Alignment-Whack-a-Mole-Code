#!/usr/bin/env python3
"""
Convert an EPUB file to plain text.

Reads the EPUB's OPF manifest and spine to determine the correct chapter
ordering, then converts each XHTML chapter to plain text via html2text.

Usage:
    python epub2txt.py <input.epub> <output.txt> --plain-text --no-metadata --ftfy

Adapted from https://github.com/soskek/bookcorpus.
"""

import argparse
import json
import os
import re
import sys
import xml.parsers.expat
import zipfile
from io import BytesIO

import html2text
from natsort import natsorted

try:
    from urllib import unquote
except ImportError:
    from urllib.parse import unquote


# ---------------------------------------------------------------------------
# EPUB XML parsers
# ---------------------------------------------------------------------------

class ContainerParser:
    """Parse META-INF/container.xml to find the root OPF file."""

    def __init__(self, xmlcontent=None):
        self.rootfile = ""
        self.xml = xmlcontent

    def startElement(self, name, attributes):
        if name == "rootfile":
            self.rootfile = attributes["full-path"]

    def parseContainer(self):
        parser = xml.parsers.expat.ParserCreate()
        parser.StartElementHandler = self.startElement
        parser.Parse(self.xml, 1)
        return self.rootfile


class BookParser:
    """Parse the OPF file to extract title, author, and NCX reference."""

    def __init__(self, xmlcontent=None):
        self.xml = xmlcontent
        self.title = ""
        self.author = ""
        self.inTitle = 0
        self.inAuthor = 0
        self.ncx = ""
        self.buffer = ""

    def startElement(self, name, attributes):
        if name == "dc:title":
            self.buffer = ""
            self.inTitle = 1
        elif name == "dc:creator":
            self.buffer = ""
            self.inAuthor = 1
        elif name == "item":
            if attributes.get("id") in ("ncx", "toc", "ncxtoc"):
                self.ncx = attributes["href"]

    def characters(self, data):
        if self.inTitle or self.inAuthor:
            self.buffer += data

    def endElement(self, name):
        if name == "dc:title":
            self.inTitle = 0
            self.title = self.buffer
            self.buffer = ""
        elif name == "dc:creator":
            self.inAuthor = 0
            self.author = self.buffer
            self.buffer = ""

    def parseBook(self):
        parser = xml.parsers.expat.ParserCreate()
        parser.StartElementHandler = self.startElement
        parser.EndElementHandler = self.endElement
        parser.CharacterDataHandler = self.characters
        parser.Parse(self.xml, 1)
        return self.title, self.author, self.ncx


class NavPoint:
    def __init__(self, id=None, playorder=None, level=0, content=None, text=None):
        self.id = id
        self.content = content
        self.playorder = playorder
        self.level = level
        self.text = text


class TocParser:
    """Parse the NCX table-of-contents file."""

    def __init__(self, xmlcontent=None):
        self.xml = xmlcontent
        self.currentNP = None
        self.stack = []
        self.inText = 0
        self.toc = []
        self.buffer = ""

    def startElement(self, name, attributes):
        if name == "navPoint":
            level = len(self.stack)
            self.currentNP = NavPoint(attributes["id"], attributes["playOrder"], level)
            self.stack.append(self.currentNP)
            self.toc.append(self.currentNP)
        elif name == "content":
            self.currentNP.content = unquote(attributes["src"])
        elif name == "text":
            self.buffer = ""
            self.inText = 1

    def characters(self, data):
        if self.inText:
            self.buffer += data

    def endElement(self, name):
        if name == "navPoint":
            self.currentNP = self.stack.pop()
        elif name == "text":
            if self.inText and self.currentNP:
                self.currentNP.text = self.buffer
            self.inText = 0

    def parseToc(self):
        parser = xml.parsers.expat.ParserCreate()
        parser.StartElementHandler = self.startElement
        parser.EndElementHandler = self.endElement
        parser.CharacterDataHandler = self.characters
        parser.Parse(self.xml, 1)
        return self.toc


# ---------------------------------------------------------------------------
# EPUB structure utilities
# ---------------------------------------------------------------------------

def _epub_name_matches(pattern, name):
    rx = re.compile(r'[^a-zA-Z_\-/.]', re.IGNORECASE)
    norm = re.sub(rx, '', name)
    return re.search(pattern, norm)


def _htmlfiles(filelist):
    return [f for f in filelist if f.endswith('htm') or f.endswith('html')]


def _uniq(xs):
    r = []
    for x in xs:
        if x not in r:
            r.append(x)
    return r


def _flatten(xs):
    r = []
    for x in xs:
        if isinstance(x, list):
            r.extend(_flatten(x))
        else:
            r.append(x)
    return r


def _string_bucket(buckets, strings, flat=False):
    strings = list(strings)
    results = []
    for bucket in buckets:
        if isinstance(bucket, str):
            bucket = bucket.split(',')
        out = []
        for pattern in bucket:
            for s in strings:
                if s not in out and _epub_name_matches(pattern, s):
                    out.append(s)
        for s in out:
            strings.remove(s)
        results.append(out)
    results.append(strings)
    if flat:
        results = _flatten(results)
    return results


def _sort_epub_files(filelist):
    *front, outro, chapters, other = _string_bucket([
        'cover', 'title', 'copyright',
        'toc,table.?of,contents',
        'frontmatter,acknowledge',
        'intro,forward',
        'index,outro,epilogue',
        '[.]htm[l]?$',
    ], natsorted(filelist), flat=False)
    return _flatten(front) + chapters + outro + other


def _rmblanklines(text):
    return '\n'.join([x for x in text.split('\n') if len(x.strip()) > 0])


def _xmlnode(element, text):
    text = re.sub(re.compile(r'<!--.*?-->', re.DOTALL), '', text)
    rx = r'<(?P<tag>{element})\s?(?P<props>.*?)(?:/>|>(?P<value>.*?)</{element}>)'.format(element=element)
    rx = re.compile(rx, re.DOTALL)
    items = [item.groupdict() for item in re.finditer(rx, text)]
    for item in items:
        props = dict(re.findall(r'([^\s]*?)="(.*?)"', item['props']))
        del item['props']
        item.update(props)
    return items


def _html_links(text):
    return [m.group(1) for m in re.finditer(r'"([^"]+?[.][a-zA-Z]{2,}(?:[#][^"]+)?)"', text)]


# ---------------------------------------------------------------------------
# OPF / manifest / spine extraction
# ---------------------------------------------------------------------------

def _extract_rootfile(file):
    filelist = [x.filename for x in file.filelist]
    if 'META-INF/container.xml' in filelist:
        result = file.read('META-INF/container.xml').decode('utf8')
        result = re.sub("='(.*?)'", r'="\1"', result)
        return result


def _extract_opf(file, meta=None):
    if meta is None:
        meta = _extract_rootfile(file)
    root = [line for line in meta.split('\n') if '<rootfile ' in line]
    assert len(root) > 0
    rootpath = _html_links(root[0])[0]
    assert rootpath.endswith('opf')
    result = file.read(rootpath).decode('utf8')
    result = re.sub("""='(.*?)'""", r'="\1"', result)
    return result


def _extract_section(name, file, opf=None):
    if opf is None:
        opf = _extract_opf(file)
    result = re.sub(
        re.compile('.*<{n}.*?>(.*?)</{n}>.*'.format(n=name), re.DOTALL),
        r'\1', opf,
    )
    return _rmblanklines(result)


def _extract_items(file, opf=None):
    manifest = _extract_section("manifest", file, opf=opf)
    return _xmlnode('item', manifest)


def _extract_spine(file, opf=None):
    spine = _extract_section("spine", file, opf=opf)
    return _xmlnode('itemref', spine)


def _href2filename(file, href, filelist, quiet=True):
    href = href.split('#', 1)[0]
    href = unquote(href)
    for name in filelist:
        if name == href or name.endswith('/' + href):
            return name
    if not quiet:
        sys.stderr.write(
            f'href2filename: failed to find href {href!r} in epub {file.filename!r}\n'
        )


def _extract_order(file, opf=None, quiet=True):
    filelist = _sort_epub_files([x.filename for x in file.filelist])
    items = _extract_items(file, opf=opf)
    spine = _extract_spine(file, opf=opf)
    ids = {x['id']: x['href'] for x in items}
    found = _uniq([_href2filename(file, ids[ref['idref']], filelist, quiet=quiet) for ref in spine])
    found = [x for x in found if x is not None]
    for filename in found:
        if filename in filelist:
            filelist.remove(filename)
    if 'META-INF/nav.xhtml' in filelist:
        filelist.remove('META-INF/nav.xhtml')
    hfiles = _htmlfiles(filelist)
    if hfiles and not quiet:
        sys.stderr.write(f'Leftover HTML files for {file.filename!r}: {hfiles!r}\n')
    return found + filelist


# ---------------------------------------------------------------------------
# Text substitution helpers
# ---------------------------------------------------------------------------

def _subst(pattern, replacement, lines, ignore=None):
    if isinstance(lines, str):
        parts = lines.split('\n')
        out = []
        for line in parts:
            if ignore is None or not re.match(ignore, line):
                line = re.sub(pattern, replacement, line)
            out.append(line)
        return '\n'.join(out)
    else:
        out = []
        for line in lines:
            if ignore is None or not re.match(ignore, line):
                line = re.sub(pattern, replacement, line)
            out.append(line)
        return out


# ---------------------------------------------------------------------------
# Main converter
# ---------------------------------------------------------------------------

class Epub2Txt:
    """Convert an EPUB file to plain text."""

    def __init__(self, epubfile, quiet=True, no_metadata=True, ftfy_fix=True,
                 plain_text=True, no_collapse_blanks=False, append_str=None):
        self.epub = epubfile if not epubfile.startswith('/dev/fd/') else BytesIO(sys.stdin.buffer.read())
        self.epub_name = epubfile
        self.quiet = quiet
        self.no_metadata = no_metadata
        self.ftfy_fix = ftfy_fix
        self.plain_text = plain_text
        self.no_collapse_blanks = no_collapse_blanks
        self.append_str = append_str

    def convert(self):
        file = zipfile.ZipFile(self.epub, "r")

        meta = _extract_rootfile(file)
        opf = _extract_opf(file, meta=meta)

        # Use OPF manifest to identify XHTML content files
        file_order = _extract_order(file, opf=opf, quiet=self.quiet)
        try:
            filelist = [x.filename for x in file.filelist]
            items = _extract_items(file, opf=opf)
            xhtml_files = set()
            for item in items:
                if (item.get('media-type') or '').lower() == 'application/xhtml+xml':
                    href = item.get('href')
                    if href:
                        resolved = _href2filename(file, href, filelist, quiet=self.quiet)
                        xhtml_files.add(resolved if resolved is not None else href)
            file_order = [x for x in file_order if x in xhtml_files]
        except Exception:
            file_order = _htmlfiles(file_order)

        files = {x: file.read(x).decode('utf-8') for x in file_order}

        content = []
        for xmlfile in file_order:
            html = files[xmlfile]
            if not self.quiet:
                sys.stderr.write(self.epub_name + '/' + xmlfile + '\n')
            h = html2text.HTML2Text()
            h.body_width = 0
            text = h.handle(html)
            if not text.endswith('\n'):
                text += '\n'
            filename = self.epub_name + '/' + xmlfile
            bookname = filename + '.md'
            if not self.no_metadata:
                content.append('<|file name={}|>'.format(json.dumps(bookname)) + '\n')
            content.append(text)
            if not self.no_metadata:
                content.append('<|/file name={}|>'.format(json.dumps(bookname)) + '\n')

        file.close()
        result = ''.join(content)

        # Fix table formatting artifacts
        result = result.replace('\n\n| \n\n', ' | ')

        if self.ftfy_fix:
            import ftfy
            result = ftfy.fix_text(result)
            result = result.replace(' …', '...')
            result = result.replace('…', '...')

        result = result.split('\n')

        ignore_ul_item = r'[*]\s'
        ignore_ol_item = r'[0-9]+[.]\s'
        ignore_li = '(?!(' + ignore_ul_item + ')|(' + ignore_ol_item + '))'
        ignore_code = '^[ ]{4,}' + ignore_li + r'[^\s]'

        def sub(pattern, replacement, text):
            return _subst(pattern, replacement, text, ignore=ignore_code)

        if self.plain_text:
            result = sub(r'[!]\s*[\[].*?[\]][(].*?[)]', ' ', result)
            result = sub(r'\[([0-9]+?)\][(].*?[)]', '', result)
            result = sub(r'[!]?\[(.*?)\][(].*?[)]', r'\1', result)

        result = sub(re.compile(r'([0-9]+)[\\][.][ ]', re.DOTALL), r'\1. ', result)
        result = '\n'.join(result)

        if not self.no_collapse_blanks:
            rx = re.compile(r'([\r\t ]*[\n]+){2,}', re.DOTALL)
            result = re.sub(rx, r'\n\n', result)

        result = re.sub(r'\n([^\n]+)[\n]#', r'\n\1\n\n#', result)

        if not self.no_collapse_blanks:
            rx = re.compile(r'([\r\t ]*[\n]+){3,}', re.DOTALL)
            result = re.sub(rx, r'\n\n\n', result)

        if self.append_str is not None:
            append = str.encode(self.append_str).decode('unicode-escape')
            result += append

        return result


def main():
    parser = argparse.ArgumentParser(
        description="Convert EPUB files to plain text.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('infile', help="Path to the input EPUB file.")
    parser.add_argument('outfile', help="Path to the output text file (use '-' for stdout).")
    parser.add_argument('-n', '--no-metadata', action="store_true",
                        help="Don't output <|file name=...|> markers.")
    parser.add_argument('-f', '--ftfy', action="store_true",
                        help="Run text through ftfy.fix_text() for Unicode cleanup.")
    parser.add_argument('-a', '--append', default=None,
                        help="Append this string to the end of the output.")
    parser.add_argument('-p', '--plain-text', action="store_true",
                        help="Strip Markdown image/link syntax to produce plain text.")
    parser.add_argument('-q', '--quiet', action="store_true",
                        help="Suppress progress messages on stderr.")
    parser.add_argument('-nc', '--no-collapse-blanks', action="store_true",
                        help="Don't collapse long runs of blank lines.")
    args = parser.parse_args()

    converter = Epub2Txt(
        args.infile,
        quiet=args.quiet,
        no_metadata=args.no_metadata,
        ftfy_fix=args.ftfy,
        plain_text=args.plain_text,
        no_collapse_blanks=args.no_collapse_blanks,
        append_str=args.append,
    )

    try:
        txt = converter.convert()
    except Exception:
        sys.stderr.write(f'Error converting {args.infile!r}:\n')
        raise

    if len(txt.strip()) > 0:
        if args.outfile == '-':
            sys.stdout.write(txt)
        else:
            with open(args.outfile, "w", encoding='utf-8') as f:
                f.write(txt)


if __name__ == "__main__":
    main()
