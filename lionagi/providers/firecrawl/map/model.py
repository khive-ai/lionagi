from pydantic import BaseModel, ConfigDict, Field


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



