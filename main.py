import json
import os
import csv
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from dotenv import load_dotenv
import requests
from openai import OpenAI

load_dotenv()

app = typer.Typer(help="People-search CLI for discovering founders, researchers, and builders.")
console = Console()

SAVED_FILE = Path.home() / ".peoplecli_saved.json"

EXA_API_KEY = os.getenv("EXA_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def load_saved() -> list[dict]:
    if SAVED_FILE.exists():
        return json.loads(SAVED_FILE.read_text())
    return []


def write_saved(people: list[dict]):
    SAVED_FILE.write_text(json.dumps(people, indent=2))


def exa_search(query: str, num_results: int = 10) -> list[dict]:
    if not EXA_API_KEY:
        console.print("[red]EXA_API_KEY not set in .env[/red]")
        raise typer.Exit(1)

    url = "https://api.exa.ai/search"
    headers = {"x-api-key": EXA_API_KEY, "Content-Type": "application/json"}
    payload = {
        "query": query,
        "numResults": num_results,
        "useAutoprompt": True,
        "contents": {"text": {"maxCharacters": 2000}},
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json().get("results", [])


def extract_people(query: str, search_results: list[dict]) -> list[dict]:
    if not OPENAI_API_KEY:
        console.print("[red]OPENAI_API_KEY not set in .env[/red]")
        raise typer.Exit(1)

    client = OpenAI(api_key=OPENAI_API_KEY)

    snippets = "\n\n---\n\n".join(
        f"URL: {r.get('url', '')}\nTitle: {r.get('title', '')}\n{r.get('text', '')[:1500]}"
        for r in search_results
    )

    prompt = f"""You are a research assistant. Given the search query and web results below, extract up to 5 real, specific people who are relevant to the query.

Search query: {query}

Web results:
{snippets}

Return a JSON array (and nothing else) where each element has exactly these keys:
- name (string)
- role (string)
- company (string)
- why_relevant (string, 1-2 sentences)
- outreach_angle (string, 1 sentence)
- source_url (string, the most relevant URL for this person)

If fewer than 5 relevant people can be identified, return only those found. Do not invent people."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    data = json.loads(raw)
    # model may return {"people": [...]} or just [...]
    if isinstance(data, list):
        return data
    for val in data.values():
        if isinstance(val, list):
            return val
    return []


def print_person(idx: int, p: dict):
    name = p.get("name", "Unknown")
    role = p.get("role", "")
    company = p.get("company", "")
    why = p.get("why_relevant", "")
    angle = p.get("outreach_angle", "")
    source = p.get("source_url", "")

    header = Text(f"{idx}. {name}", style="bold cyan")
    body = (
        f"[bold]Role:[/bold] {role}\n"
        f"[bold]Company:[/bold] {company}\n"
        f"[bold]Why relevant:[/bold] {why}\n"
        f"[bold]Outreach angle:[/bold] {angle}\n"
        f"[bold]Source:[/bold] [link={source}]{source}[/link]"
    )
    console.print(Panel(body, title=header, expand=False))


@app.command()
def find(
    query: str = typer.Argument(..., help='E.g. "AI healthcare founders using NVIDIA"'),
    results: int = typer.Option(10, "--results", "-n", help="Number of web results to fetch"),
):
    """Search for people matching a query and display structured profiles."""
    console.print(f"[dim]Searching for:[/dim] [bold]{query}[/bold]\n")

    with console.status("Fetching web results via Exa…"):
        search_results = exa_search(query, results)

    if not search_results:
        console.print("[yellow]No web results found.[/yellow]")
        raise typer.Exit()

    with console.status("Extracting people with GPT-4o-mini…"):
        people = extract_people(query, search_results)

    if not people:
        console.print("[yellow]Could not identify specific people from the results.[/yellow]")
        raise typer.Exit()

    console.print(f"[green]Found {len(people)} people:[/green]\n")
    for i, p in enumerate(people, 1):
        print_person(i, p)


@app.command()
def save(name: str = typer.Argument(..., help="Full name of the person to save")):
    """Save a person by name (must have been found recently via `find`)."""
    # Since we don't persist find results between runs, ask user for details manually
    console.print(f"[dim]Saving:[/dim] [bold]{name}[/bold]")
    role = typer.prompt("Role")
    company = typer.prompt("Company")
    why = typer.prompt("Why relevant")
    angle = typer.prompt("Outreach angle")
    source = typer.prompt("Source URL", default="")

    person = {
        "name": name,
        "role": role,
        "company": company,
        "why_relevant": why,
        "outreach_angle": angle,
        "source_url": source,
    }

    saved = load_saved()
    # avoid duplicates by name
    if any(p["name"].lower() == name.lower() for p in saved):
        console.print(f"[yellow]{name} is already saved.[/yellow]")
        raise typer.Exit()

    saved.append(person)
    write_saved(saved)
    console.print(f"[green]Saved {name}.[/green]")


@app.command(name="list")
def list_people():
    """List all saved people."""
    saved = load_saved()
    if not saved:
        console.print("[yellow]No saved people yet. Use `find` then `save`.[/yellow]")
        raise typer.Exit()

    console.print(f"[green]{len(saved)} saved people:[/green]\n")
    for i, p in enumerate(saved, 1):
        print_person(i, p)


@app.command()
def export(
    output: str = typer.Option("people.csv", "--output", "-o", help="Output CSV file path"),
):
    """Export saved people to a CSV file."""
    saved = load_saved()
    if not saved:
        console.print("[yellow]No saved people to export.[/yellow]")
        raise typer.Exit()

    fields = ["name", "role", "company", "why_relevant", "outreach_angle", "source_url"]
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for p in saved:
            writer.writerow({k: p.get(k, "") for k in fields})

    console.print(f"[green]Exported {len(saved)} people to {output}[/green]")


@app.command()
def intro(name: str = typer.Argument(..., help="Name of the saved person to draft outreach for")):
    """Generate a cold outreach message for a saved person."""
    saved = load_saved()
    match = next((p for p in saved if p["name"].lower() == name.lower()), None)

    if not match:
        console.print(f"[red]{name} not found in saved people. Run `save` first.[/red]")
        raise typer.Exit(1)

    if not OPENAI_API_KEY:
        console.print("[red]OPENAI_API_KEY not set in .env[/red]")
        raise typer.Exit(1)

    client = OpenAI(api_key=OPENAI_API_KEY)

    prompt = f"""Write a short, genuine cold outreach message (3-4 sentences) for the following person.
Do not be sycophantic. Be direct and specific about why you're reaching out.

Name: {match['name']}
Role: {match['role']}
Company: {match['company']}
Why relevant: {match['why_relevant']}
Outreach angle: {match['outreach_angle']}

Return only the message text."""

    with console.status("Drafting outreach message…"):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )

    message = response.choices[0].message.content.strip()
    console.print(Panel(message, title=f"[bold cyan]Outreach: {match['name']}[/bold cyan]", expand=False))


if __name__ == "__main__":
    app()
