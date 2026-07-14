import sqlite3
from typing import Optional
import strawberry
from fastapi import FastAPI
from strawberry.fastapi import GraphQLRouter

# The @strawberry.federation.type decorator is the magic here.
# By specifying keys=["id"], we tell the future Supergraph: 
# "If you ever need to merge my data with another API, use the 'id' field to link us."
@strawberry.federation.type(keys=["id"])
class Movie:
    id: int
    title: Optional[str] = None
    release_date: Optional[str] = None
    release_year: Optional[int] = None
    genre: Optional[str] = None
    runtime: Optional[int] = None
    overview: Optional[str] = None

    @classmethod
    def resolve_reference(cls, id: int):
        """
        This method is called automatically by the Supergraph when it needs 
        to fetch Content data for a specific Movie ID.
        """
        with sqlite3.connect("content.db") as conn:
            conn.row_factory = sqlite3.Row  # Return rows as dictionaries
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM movies WHERE id = ?", (id,))
            row = cursor.fetchone()
            
            if row:
                return cls(**dict(row))
            return None

@strawberry.type
class Query:
    # A basic query to test the API directly (outside of federation)
    @strawberry.field
    def movie_content(self, id: int) -> Optional[Movie]:
        return Movie.resolve_reference(id)

# We must enable federation 2 explicitly for Apollo Router compatibility.
schema = strawberry.federation.Schema(query=Query, federation_version="2.0")
graphql_app = GraphQLRouter(schema)

app = FastAPI(title="Netflix Content Subgraph")
app.include_router(graphql_app, prefix="/graphql")