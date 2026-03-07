from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class OutputFormat(str, Enum):
    markdown = "markdown"
    html = "html"
    raw_html = "rawHtml"
    links = "links"
    screenshot = "screenshot"


class FirecrawlScrapeRequest(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        serialize_by_alias=True,
    )

    url: str = Field(..., description="The URL to scrape.")
    formats: list[OutputFormat] | None = Field(
        default=None,
        description="Output formats to return (markdown, html, rawHtml, links, screenshot).",
    )
    only_main_content: bool | None = Field(
        default=True,
        alias="onlyMainContent",
        description="Only return the main content of the page, excluding navs/footers.",
    )
    include_tags: list[str] | None = Field(
        default=None,
        alias="includeTags",
        description="HTML tags to include in extraction.",
    )
    exclude_tags: list[str] | None = Field(
        default=None,
        alias="excludeTags",
        description="HTML tags to exclude from extraction.",
    )
    wait_for: int | None = Field(
        default=None,
        alias="waitFor",
        description="Milliseconds to wait before scraping (for JS-rendered pages).",
    )
    timeout: int | None = Field(
        default=None,
        description="Timeout in milliseconds for the scrape request.",
    )


class FirecrawlMapRequest(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        serialize_by_alias=True,
    )

    url: str = Field(..., description="The base URL to map.")
    search: str | None = Field(
        default=None,
        description="Optional search term to filter mapped URLs.",
    )
    ignore_sitemap: bool | None = Field(
        default=None,
        alias="ignoreSitemap",
        description="Ignore the website sitemap when crawling.",
    )
    include_subdomains: bool | None = Field(
        default=None,
        alias="includeSubdomains",
        description="Include subdomains of the website.",
    )
    limit: int | None = Field(
        default=None,
        description="Maximum number of links to return.",
    )
