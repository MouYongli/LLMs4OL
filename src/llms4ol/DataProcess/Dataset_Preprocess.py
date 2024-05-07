#from datasets import load_dataset

#raw_dataset = load_dataset("wikipedia", language="en", date="20240501")
#print(raw_dataset[:10])
import mwparserfromhell
import requests
import json
import os
import re

def type_definition_from_wiki(type):

    API_URL = "https://en.wikipedia.org/w/api.php"

    def parse(title):
        params = {
            "action": "query",
            "prop": "revisions",
            "rvprop": "content",
            "rvslots": "main",
            "rvlimit": 1,
            "titles": title,
            "format": "json",
            "formatversion": "2",
        }
        headers = {"User-Agent": "My-Bot-Name/1.0"}
        req = requests.get(API_URL, headers=headers, params=params)
        res = req.json()
        revision = res["query"]["pages"][0]["revisions"][0]
        text = revision["slots"]["main"]["content"]
        return mwparserfromhell.parse(text)
    
    wikicode = parse(type)
    # Filters for magic words that are parser instructions -- e.g., __NOTOC__
    re_rm_magic = re.compile("__[A-Z]*__", flags=re.UNICODE)

    # Filters for file/image links.
    media_prefixes = "|".join(["File", "Image", "Media"])
    re_rm_wikilink = re.compile(f"^(?:{media_prefixes}):", flags=re.IGNORECASE | re.UNICODE)

    def rm_wikilink(obj):
        return bool(re_rm_wikilink.match(str(obj.title)))

    # Filters for references and tables
    def rm_tag(obj):
        return str(obj.tag) in {"ref", "table"}

    # Leave category links in-place but remove the category prefixes
    cat_prefixes = "|".join(["Category"])
    re_clean_wikilink = re.compile(f"^(?:{cat_prefixes}):", flags=re.IGNORECASE | re.UNICODE)

    def is_category(obj):
        return bool(re_clean_wikilink.match(str(obj.title)))

    def clean_wikilink(obj):
        text = obj.__strip__()
        text = re.sub(re_clean_wikilink, "", text)
        obj.text = text

    def try_replace_obj(obj):
        try:
            clean_wikilink(obj)
        except ValueError:
            # For unknown reasons, objects are sometimes not found.
            pass

    def try_remove_obj(obj, section):
        try:
            section.remove(obj)
        except ValueError:
            # For unknown reasons, objects are sometimes not found.
            pass

    section_text = []
    # Filter individual sections to clean.
    for section in wikicode.get_sections(flat=True, include_lead=True, include_headings=True):
        for obj in section.ifilter_wikilinks(recursive=True):
            if rm_wikilink(obj):
                try_remove_obj(obj, section)
            elif is_category(obj):
                try_replace_obj(obj)
        for obj in section.ifilter_tags(matches=rm_tag, recursive=True):
            try_remove_obj(obj, section)
        content = re.sub(re_rm_magic, "", section.strip_code().strip()).replace("\n", "").replace("()","")
        if "See also" in content:
            break
        else:
            section_text.append(content)
    return section_text[0]

print(type_definition_from_wiki("levee"))