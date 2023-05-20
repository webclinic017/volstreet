from discord.ext import commands
import discord
import json

prefix = "!"
intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix=prefix, intents=intents)


# Add the indices list as an argument to the run_bot function
def run_bot(token, indices, user_id_map=None):
    # Set the attributes on the bot object
    if user_id_map is None:
        user_id_map = {}
    bot.indices = indices
    bot.user_id_map = user_id_map
    bot.run(token)


# Modify the find_index_by_name function to accept the indices list
def find_index_by_name(index_name, indices):
    for index in indices:
        if index.name == index_name:
            return index
    return None


@bot.command(name="exit")
async def exit_strats(ctx, *exit_indices):
    for index in exit_indices:
        # Pass the bot.indices list to the find_index_by_name function
        e_index = find_index_by_name(index, bot.indices)
        if e_index is not None and e_index.traded:
            e_index.intraday_straddle_forced_exit = True
            await ctx.send(f"Exiting {e_index.name}")
        else:
            await ctx.send(f"Index {index} not found or not traded.")


@bot.command(name="exitnew")
async def exit_strats(ctx, *exit_indices):
    user_id = str(ctx.message.author.id)
    client_code = bot.user_id_map.get(user_id, None)
    if client_code is None:
        await ctx.send(f"User {user_id} not found")
        return
    for index in exit_indices:
        with open(f'{client_code}_{index}_force_exit.json', "w") as file:
            json.dump(True, file)
        await ctx.send(f"Exiting {index}")
