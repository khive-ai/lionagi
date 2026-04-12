from lionagi import Branch


async def try_codex():
    branch = Branch(
        chat_model="codex/gpt-5.3-codex-spark",
        system_datetime=True,
        system="You are a diligent codebase investigator",
    )

    response = await branch.chat(
        instruction="please examine the repo, what is it? Is it a good codebase? Rate 1-10 with reasoning",
        guidance="read into key files, do not take docs at face value",
        verbose_output=True,
        # reasoning_effort="medium",
    )
    print(response)


if __name__ == "__main__":
    import anyio

    anyio.run(try_codex)
