import arxiv
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import weave

class Author(BaseModel):
    full_name: str

class Link(BaseModel):
    href: str
    title: Optional[str] = None
    rel: Optional[str] = None
    content_type: Optional[str] = None

class ArxivPaper(BaseModel):
    entry_id: str
    updated: datetime
    published: datetime
    title: str
    authors: List[Author]
    summary: str
    comment: Optional[str] = None
    journal_ref: Optional[str] = None
    doi: Optional[str] = None
    primary_category: str
    categories: List[str]
    links: List[Link]
    pdf_url: Optional[str] = None

    def __getitem__(self, key):
        return getattr(self, key)

def convert_raw_arxiv_to_pydantic(paper):
    return ArxivPaper(
        entry_id=paper.entry_id,
        updated=paper.updated,
        published=paper.published,
        title=paper.title,
        authors=[Author(full_name=str(author)) for author in paper.authors],
        summary=paper.summary,
        comment=paper.comment,
        journal_ref=paper.journal_ref,
        doi=paper.doi,
        primary_category=paper.primary_category,
        categories=paper.categories,
        links=[Link(href=link.href, title=link.title, rel=link.rel, content_type=link.content_type) 
                for link in paper.links],
        pdf_url=paper.pdf_url
    )
