import arxiv
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

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

    @property
    def pdf_url(self):
        """
        Finds the PDF link among a result's links and returns its URL.

        Should only be called once for a given `Result`, in its constructor.
        After construction, the URL should be available in `Result.pdf_url`.
        """
        pdf_urls = [link.href for link in self.links if link.title == "pdf"]
        if len(pdf_urls) == 0:
            return None
        elif len(pdf_urls) > 1:
            print("Result has multiple PDF links; using %s", pdf_urls[0])
        return pdf_urls[0]

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
                for link in paper.links]
    )