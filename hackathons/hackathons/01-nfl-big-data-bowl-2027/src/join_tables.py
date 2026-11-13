"""Schema-validated join utilities for NFL Big Data Bowl 2027 data."""

import pandas as pd
import numpy as np


def validate_nfl_id(df, col='nfl_id'):
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame")
    df = df.copy()
    df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
    nulls = df[col].isna().sum()
    if nulls > 0:
        import warnings
        warnings.warn(f"{nulls} null nfl_id values in {col}")
    return df


def validate_game_id(df, col='game_id'):
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame")
    df = df.copy()
    df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
    return df


def validate_play_id(df, col='play_id'):
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame")
    df = df.copy()
    df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
    return df


def join_combine_results_players(combine_results, players, validate=True):
    if validate:
        combine_results = validate_nfl_id(combine_results)
        players = validate_nfl_id(players)
    combine_results = combine_results.astype({'nfl_id': 'int64'})
    players = players.astype({'nfl_id': 'int64'})
    merged = pd.merge(combine_results, players, on='nfl_id', how='left', validate='many_to_one')
    merged = merged.convert_dtypes()
    return merged


def join_combine_results_tracking(combine_results, combine_tracking, validate=True):
    if validate:
        combine_results = validate_nfl_id(combine_results)
        combine_tracking = validate_nfl_id(combine_tracking)
    combine_results = combine_results.astype({'nfl_id': 'int64'})
    combine_tracking = combine_tracking.astype({'nfl_id': 'int64'})
    merged = pd.merge(combine_results, combine_tracking, on='nfl_id', how='left', validate='many_to_many')
    merged = merged.convert_dtypes()
    return merged


def join_players_combine_results(players, combine_results, validate=True):
    if validate:
        players = validate_nfl_id(players)
        combine_results = validate_nfl_id(combine_results, 'nfl_id')
    players = players.astype({'nfl_id': 'int64'})
    combine_results = combine_results.astype({'nfl_id': 'int64'})
    merged = pd.merge(players, combine_results, on='nfl_id', how='left', validate='many_to_one')
    merged = merged.convert_dtypes()
    return merged


def join_player_play_games(player_play, games, validate=True):
    if validate:
        player_play = validate_game_id(player_play, 'game_id')
        games = validate_game_id(games, 'game_id')
    player_play = player_play.astype({'game_id': 'int64'})
    games = games.astype({'game_id': 'int64'})
    merged = pd.merge(player_play, games, on='game_id', how='left', validate='many_to_one')
    merged = merged.convert_dtypes()
    return merged


def build_full_pipeline(combine_results, combine_tracking, players, player_career_successes,
                        player_play, game_tracking_2023, games, chunksize=50000):
    combine_results = validate_nfl_id(combine_results)
    combine_tracking = validate_nfl_id(combine_tracking)
    players = validate_nfl_id(players)
    player_career_successes = validate_nfl_id(player_career_successes)
    player_play = validate_nfl_id(player_play, 'nfl_id')
    games = validate_game_id(games)
    game_tracking_2023 = validate_nfl_id(game_tracking_2023, 'nfl_id')
    
    cr = join_combine_results_players(combine_results, players)
    crt = join_combine_results_tracking(combine_results, combine_tracking)
    pp = pd.merge(player_play, players, on='nfl_id', how='left', validate='many_to_one').convert_dtypes()
    ppcr = pd.merge(player_play, combine_results, on='nfl_id', how='left', validate='many_to_many').convert_dtypes()
    ppcs = pd.merge(player_play, player_career_successes, on='nfl_id', how='left', validate='many_to_one').convert_dtypes()
    ppgs = pd.merge(player_play, games, on='game_id', how='left', validate='many_to_one').convert_dtypes()
    
    master = ppcr.merge(ppcs, on=['game_id', 'play_id', 'nfl_id'], how='left').merge(
        ppgs, on=['game_id', 'play_id'], how='left').merge(games, on='game_id', how='left')
    
    for c in ['draft_year', 'combine_position', 'forty', 'vertical', 'three_cone',
              'short_shuttle', 'broad_jump', 'ten_yd_split']:
        if c not in master.columns:
            master[c] = np.nan
    
    col_order = [
        'game_id', 'play_id', 'nfl_id',
        'route_ran', 'receiver_alignment', 'pass_result', 'separation_at_pass_forward',
        'cushion', 'target', 'rec_yards', 'yards_after_catch',
        'combine_position', 'forty', 'vertical', 'three_cone', 'short_shuttle',
        'broad_jump', 'ten_yd_split',
        'career_games_active', 'career_games_started',
        'ap_all_pro_1st_team', 'ap_all_pro_2nd_team', 'pro_bowl_original_ballot',
        'display_name', 'draft_year', 'nfl_position', 'draft_round',
        'draft_pick_within_round', 'draft_overall_pick',
        'quarter', 'down', 'yards_to_go', 'play_direction',
        'possession_team', 'yardline_side', 'yardline_number',
        'mean_speed', 'mean_accel', 'max_speed', 'max_accel', 'n_tracking_points',
        'yards_gained', 'pass_result',
    ]
    existing_order = [c for c in col_order if c in master.columns]
    master = master[existing_order]
    
    return {
        'combine_results': cr,
        'combine_tracking': combine_tracking,
        'players': players,
        'player_career_successes': player_career_successes,
        'player_play': player_play,
        'games': games,
        'game_tracking_2023': game_tracking_2023,
        'master': master,
        'player_play_with_combine': ppcr,
        'player_play_with_players': pp,
        'player_play_with_career': ppcs,
    }
