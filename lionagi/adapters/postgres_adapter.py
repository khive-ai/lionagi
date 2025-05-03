from .sql_adapter import SQLAdapter


class PostgresAdapter(SQLAdapter):
    obj_key = "postgres"

    DEFAULT_DSN = "postgresql+psycopg://user:pass@localhost:5432/db"

    @classmethod
    def from_obj(cls, subj_cls, obj: dict, /, **kw):
        obj = {"engine_url": obj.get("engine_url", cls.DEFAULT_DSN), **obj}
        return super().from_obj(subj_cls, obj, **kw)

    @classmethod
    def to_obj(cls, subj, /, **kw):
        kw.setdefault("engine_url", cls.DEFAULT_DSN)
        return super().to_obj(subj, **kw)
