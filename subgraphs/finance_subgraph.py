import sqlite3
from typing import NewType, Optional
import strawberry
from fastapi import FastAPI
from strawberry.fastapi import GraphQLRouter

BigInt = strawberry.scalar(
    NewType("BigInt", int),
    name="BigInt",
    description="A 64-bit or larger integer serialized as a string",
    serialize=lambda v: str(v),
    parse_value=lambda v: int(v),
)

# Notice how the Finance team's 'Movie' only contains money fields.
# They know nothing about titles or genres, but they share the same keys=["id"] hook!
@strawberry.federation.type(keys=["id"])
class Movie:
    id: int
    budget: Optional[BigInt] = None
    revenue: Optional[BigInt] = None

    @classmethod
    def resolve_reference(cls, id: int):
        """
        When the Supergraph asks for budget/revenue for a Movie ID,
        this resolver digs into the isolated finance.db to find it.
        """
        with sqlite3.connect("finance.db") as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            # Our table in finance.db is named 'financials' based on our Phase 1 script
            cursor.execute("SELECT * FROM financials WHERE id = ?", (id,))
            row = cursor.fetchone()
            
            if row:
                return cls(**dict(row))
            
            # If a movie isn't found in financials, return 0s instead of crashing
            return cls(id=id, budget=0, revenue=0)

@strawberry.type
class Query:
    @strawberry.field
    def movie_finance(self, id: int) -> Optional[Movie]:
        return Movie.resolve_reference(id)

schema = strawberry.federation.Schema(query=Query, federation_version="2.0")
graphql_app = GraphQLRouter(schema)

app = FastAPI(title="Netflix Finance Subgraph")
app.include_router(graphql_app, prefix="/graphql")