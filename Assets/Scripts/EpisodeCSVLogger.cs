using System;
using System.IO;
using UnityEngine;

public static class EpisodeCSVLogger
{
    // Once per unity run
    private static string filePathAgentEpisodes;
    private static string filePathGlobalEpisodes;
    private static bool initialized = false;

    private static string STATS_DIRECTORY = "/EvaluationResults";

    public static void Initialize(string runName)
    {
        if (initialized)
            return;

        // In order to avoid overwriting the previous files, use a timestamp in file name
        string timestamp = System.DateTime.Now.ToString("yyyyMMdd_HHmmss");

        // Create the directory where the csv files will be stored if it does not exist
        string directory = Application.dataPath + $"{STATS_DIRECTORY}/{runName}_{timestamp}/";
        if (!Directory.Exists(directory))
            Directory.CreateDirectory(directory);

        filePathAgentEpisodes = Path.Combine(directory, "agents.csv");
        filePathGlobalEpisodes = Path.Combine(directory, "global.csv");

        // Open the agent episode file
        using (StreamWriter writer = new StreamWriter(filePathAgentEpisodes, false))
        {
            writer.WriteLine(GetAgentEpisodeHeader());
        }

        using (StreamWriter writer = new StreamWriter(filePathGlobalEpisodes, false))
        {
            writer.WriteLine(GetGlobalEpisodeHeader());
        }

        initialized = true;
    }

    public static string GetAgentEpisodeHeader()
    {
        return "episode;personality;hitsInflicted;dragonsKilled;hasKey;timeToFindKey;agentCollisions;distanceTraveled;avgSpeed;avgDistanceFromAgents;avgDistanceFromDragons;idleTime;timeInVicinityOfAgents";
    }

    public static string GetGlobalEpisodeHeader()
    {
        return "episode;win;failReason;episodeDuration;timeToKillAllDragons;timeToGrabKey;timeFromKeyGrabToEscape";
    }

    public static void LogAgentEpisode(string row)
    {
        using (StreamWriter writer = new StreamWriter(filePathAgentEpisodes, true))
        {
            writer.WriteLine(row);
        }
    }

    public static void LogGlobalEpisode(string row)
    {
        using (StreamWriter writer = new StreamWriter(filePathGlobalEpisodes, true))
        {
            writer.WriteLine(row);
        }
    }
}
