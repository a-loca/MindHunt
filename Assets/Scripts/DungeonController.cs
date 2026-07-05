using System.Collections.Generic;
using System.Linq;
using Unity.Barracuda;
using Unity.MLAgents;
using Unity.MLAgents.Policies;
using UnityEngine;

public class DungeonController : MonoBehaviour
{
    // ========================================================================
    // Inspector fields
    // ========================================================================
    [Header("Stats")]
    public bool computeEpisodeStats = false;

    [Header("Environment Objects")]
    [SerializeField]
    private GameObject cave;

    [SerializeField]
    private GameObject door;

    [SerializeField]
    private GameObject floor;

    [SerializeField]
    private GameObject columns;

    [Header("Prefabs")]
    [SerializeField]
    private GameObject agentPrefab;

    [SerializeField]
    private GameObject dragonPrefab;

    [SerializeField]
    private GameObject keyPrefab;

    [Header("Episode Settings")]
    public int numberOfAgents = 3;
    public int numberOfDragons = 2;

    [Header("Timer")]
    public float timeToEscape = 30f;

    [Header("Reward System")]
    [SerializeField]
    private GroupRewardSystem groupRewardSystem;

    // Need an object of both personality and behavior name
    [Header("Personality")]
    public bool inference = false;

    [System.Serializable]
    public struct PersonalitySettings
    {
        public Personality personality;
        public string behaviorName;
        public NNModel inferenceModel;
    }

    public PersonalitySettings[] personalitySettings;

    // ========================================================================
    // Public state information of the dungeon, read by agents or needed for stats
    // ========================================================================
    [HideInInspector]
    public float maxDistance;

    [HideInInspector]
    public List<DragonBehavior> dragons;

    [HideInInspector]
    public List<AgentBehavior> agents;

    [HideInInspector]
    public bool keyGrabbedByAgent = false;

    [HideInInspector]
    public int remainingDragons;

    [HideInInspector]
    public Timer timer;

    [HideInInspector]
    public GameObject key;

    [HideInInspector]
    public float urgency = 0f;

    // ========================================================================
    // Private state information
    // ========================================================================
    private int episodeCounter = 0;
    private DoorController doorController;
    private Bounds floorBounds;
    private SimpleMultiAgentGroup agentGroup;
    private GlobalEpisodeStats globalEpisodeStats;
    private LayerMask spawnBlockerLayers;

    // ========================================================================
    // Constants
    // ========================================================================
    private const float CAVE_MARGIN_FROM_WALL = 0.7f;
    private const float MARGIN_FROM_WALL = 1f;
    private const float SPAWN_HEIGHT_OFFSET = 0.3f;
    private const float CAVE_SPAWN_HEIGHT_OFFSET = 0.5f;
    private const float DOOR_SPAWN_HEIGHT_OFFSET = 0.9f;
    private const float SAFE_SPAWN_RADIUS = 0.8f;

    private enum LightColor
    {
        Default,
        Green,
        Red,
    };

    private readonly Dictionary<LightColor, string> LightColorToHex = new()
    {
        { LightColor.Green, "#3AC186" },
        { LightColor.Red, "#C13A61" },
        { LightColor.Default, "#943AC1" },
    };

    // ========================================================================
    // Initialization and reset
    // ========================================================================
    void Start()
    {
        spawnBlockerLayers = LayerMask.GetMask("Obstacle", "Door", "Dragon", "Agent", "Key");

        agents = new List<AgentBehavior>();
        dragons = new List<DragonBehavior>();
        agentGroup = new SimpleMultiAgentGroup();

        floorBounds = floor.GetComponent<Collider>().bounds;
        maxDistance = Mathf.Sqrt(
            floorBounds.size.x * floorBounds.size.x + floorBounds.size.z * floorBounds.size.z
        );

        InstantiateDragons();
        InstantiateAgents(); // this also registers them into the group

        timer = GetComponent<Timer>();
        timer.onTimerEndEvent += () => FailEpisode(FailureReason.Timer);

        doorController = door.GetComponent<DoorController>();
        doorController.OnAgentEscape += WinEpisode;

        if (computeEpisodeStats)
        {
            string teamName = GetTeamName();

            EpisodeCSVLogger.Initialize(teamName);
        }

        ResetEnvironment();
    }

    public void ResetEnvironment()
    {
        // Save global episode stats and create a new object for the next episode
        if (episodeCounter > 0 && computeEpisodeStats)
        {
            EpisodeCSVLogger.LogGlobalEpisode(globalEpisodeStats.ToCSVString(episodeCounter));
            Debug.Log("Completed episode: " + episodeCounter);
        }

        // Reset the stats
        if (computeEpisodeStats)
            globalEpisodeStats = new GlobalEpisodeStats(this);

        // Stop running timer from previous episode
        timer.StopTimer();

        // Destroy key if it has not been picked up
        if (key != null)
            Destroy(key);

        keyGrabbedByAgent = false;

        // Reset number of dragons in the environment
        remainingDragons = numberOfDragons;

        // Reset urgency
        urgency = 0f;

        // Reposition and reset agents
        foreach (AgentBehavior agent in agents)
        {
            // Log stats of the episode that just ended
            if (episodeCounter > 0 && computeEpisodeStats)
                EpisodeCSVLogger.LogAgentEpisode(agent.stats.ToCSVString(episodeCounter));

            agent.Reset();

            // Reposition the agent
            agent.transform.position = GetRandomPosition();

            // Reset rotation to look in a random direction
            agent.transform.rotation = Quaternion.Euler(0, Random.Range(0, 360), 0);
        }

        // Respawn door
        SpawnDoor();

        // Lock door
        doorController.LockDoor();

        // Respawn cave
        Vector3[] dragonPositions = SpawnCaveAndDragons();

        // Heal and reposition dragon (need to warp the mesh agent)
        for (int i = 0; i < dragons.Count; i++)
        {
            dragons[i].Resuscitate(dragonPositions[i]);
        }

        // Increment episode counter
        episodeCounter++;

        //Debug.Log("Starting episode " + episodeCounter);
    }

    // ========================================================================
    // Episode handling
    // ========================================================================
    public void DragonWasKilled()
    {
        remainingDragons--;

        agentGroup.AddGroupReward(groupRewardSystem.killDragon);

        if (remainingDragons == 0)
        {
            // Debug.Log("All dragons have been slain!");

            // Start the timer to escape
            timer.StartTimer(timeToEscape);

            // All agents need to know that the dragons are dead
            foreach (AgentBehavior agent in agents)
            {
                agent.AreDragonsAlive = false;
            }

            // Spawn the key in the environment
            InstantiateKey();
        }
    }

    public void FailEpisode(FailureReason reason)
    {
        //Debug.Log("Agents failed to escape.");

        // Record the loss and the reason for stats
        if (computeEpisodeStats)
        {
            globalEpisodeStats.failReason = reason;
            globalEpisodeStats.win = false;
        }

        // The group gets a punishment
        agentGroup.AddGroupReward(groupRewardSystem.dragonsEscape);

        ChangeLightsColor(LightColor.Red);

        // End group episode
        agentGroup.EndGroupEpisode();
        ResetEnvironment();
    }

    private void WinEpisode()
    {
        // Debug.Log("Door was unlocked, agents escaped!");

        // Record the win in the stats
        if (computeEpisodeStats)
            globalEpisodeStats.win = true;

        ChangeLightsColor(LightColor.Green);

        // Group reward
        agentGroup.AddGroupReward(groupRewardSystem.escape);

        // End the episode
        agentGroup.EndGroupEpisode();

        // Prepare for next episode
        ResetEnvironment();
    }

    // ========================================================================
    // Instantiate objects from prefabs
    // ========================================================================
    private void InstantiateAgents()
    {
        for (int i = 0; i < numberOfAgents; i++)
        {
            GameObject agent = Instantiate(agentPrefab, transform);

            AgentBehavior agentBehavior = agent.GetComponent<AgentBehavior>();

            agentBehavior.computeEpisodeStats = computeEpisodeStats;

            agentBehavior.Id = i;

            agentBehavior.Dungeon = this;

            // Set the OCEAN personality parameters of the agent
            agentBehavior.Personality = personalitySettings[i].personality;

            // Set the correct behavior name in order to train the correct model
            BehaviorParameters behaviorParams = agent.GetComponent<BehaviorParameters>();
            behaviorParams.BehaviorName = personalitySettings[i].behaviorName;
            // Set the correct mode: training or inference
            behaviorParams.Model = personalitySettings[i].inferenceModel;
            behaviorParams.BehaviorType = inference
                ? BehaviorType.InferenceOnly
                : BehaviorType.Default;

            // Add to the collection of agents spawned by the environment
            agents.Add(agentBehavior);

            // Register the agent to the multiagent group
            agentGroup.RegisterAgent(agentBehavior);
        }
    }

    private void InstantiateDragons()
    {
        for (int i = 0; i < numberOfDragons; i++)
        {
            GameObject dragon = Instantiate(dragonPrefab, transform);

            DragonBehavior db = dragon.GetComponent<DragonBehavior>();

            db.SetCave(cave);

            db.onDragonEscapeEvent += () => FailEpisode(FailureReason.DragonEscaped);
            db.onDragonSlainEvent += DragonWasKilled;

            dragons.Add(db);
        }
    }

    private void InstantiateKey()
    {
        // Spawn key in random position
        GameObject key = Instantiate(
            keyPrefab,
            GetRandomPosition(),
            Quaternion.identity,
            transform
        );

        this.key = key;
    }

    public void KeyWasGrabbed()
    {
        // Destroy the game object of the key in the environment
        Destroy(key);

        keyGrabbedByAgent = true;

        agentGroup.AddGroupReward(groupRewardSystem.grabKey);
    }

    // ========================================================================
    // Positionin the objects in the environment
    // ========================================================================
    private void SpawnDoor()
    {
        // Spawn the door in one of the 4 sides of the arena
        float margin = MARGIN_FROM_WALL;

        float y = floorBounds.max.y + DOOR_SPAWN_HEIGHT_OFFSET;
        Vector3 center = floorBounds.center;

        // Defining the 4 possible spawn points
        // which are the center point of each wall
        Vector3[] sideCenters =
        {
            new Vector3(floorBounds.min.x + margin, y, center.z), // left side
            new Vector3(floorBounds.max.x - margin, y, center.z), // right side
            new Vector3(center.x, y, floorBounds.min.z + margin), // bottom side
            new Vector3(center.x, y, floorBounds.max.z - margin), // top side
        };

        // Extracting a random side
        door.transform.position = sideCenters[Random.Range(0, sideCenters.Length)];

        // Rotate door towards center of the floor
        door.transform.LookAt(new Vector3(center.x, center.y + DOOR_SPAWN_HEIGHT_OFFSET, center.z));
    }

    private Vector3[] SpawnCaveAndDragons()
    {
        float margin = CAVE_MARGIN_FROM_WALL;

        float cave_y = floorBounds.max.y + CAVE_SPAWN_HEIGHT_OFFSET;

        // Define the 4 corners of the arena
        Vector3[] corners =
        {
            new Vector3(floorBounds.min.x + margin, cave_y, floorBounds.min.z + margin),
            new Vector3(floorBounds.min.x + margin, cave_y, floorBounds.max.z - margin),
            new Vector3(floorBounds.max.x - margin, cave_y, floorBounds.min.z + margin),
            new Vector3(floorBounds.max.x - margin, cave_y, floorBounds.max.z - margin),
        };

        // Choose a random corner of the 4 available on the floor
        Vector3 cavePosition = corners[Random.Range(0, corners.Length)];

        // Move cave to generated position
        cave.transform.position = cavePosition;

        // Rotate cave towards center of the floor
        cave.transform.LookAt(floorBounds.center);

        // Check where the cave is
        bool caveOnLeft = cavePosition.x < floorBounds.center.x;
        bool caveOnBottom = cavePosition.z < floorBounds.center.z;

        // Find bounds of the opposite half of the dungeon
        // with respect to the cave.

        // If the cave is on the left, then the lower bound of the
        // spawn area of the dragon should be the center,
        // and the top the top right. If the cave is on the right, then
        // the upper bound is the center of the arena, the lower bound is
        // the opposite corner
        float minX = caveOnLeft ? floorBounds.center.x : floorBounds.min.x + margin;
        float maxX = caveOnLeft ? floorBounds.max.x - margin : floorBounds.center.x;

        // If the cave is on the bottom side,
        // lower bound is the center, upper bound is the top of the arena
        float minZ = caveOnBottom ? floorBounds.center.z : floorBounds.min.z + margin;
        float maxZ = caveOnBottom ? floorBounds.max.z - margin : floorBounds.center.z;

        return GetNonOverlappingRandomPositions(numberOfDragons, minX, maxX, minZ, maxZ);
    }

    private Vector3 GetRandomPosition()
    {
        // Position slightly on top of the floor
        float y = floorBounds.max.y + SPAWN_HEIGHT_OFFSET;

        // Some margin to avoid spawning on the walls
        float margin = MARGIN_FROM_WALL;

        float minX = floorBounds.min.x + margin;
        float maxX = floorBounds.max.x - margin;
        float minZ = floorBounds.min.z + margin;
        float maxZ = floorBounds.max.z - margin;

        return GetNonOverlappingRandomPositions(1, minX, maxX, minZ, maxZ)[0];
    }

    private Vector3[] GetNonOverlappingRandomPositions(
        int numberOfPositions,
        float minX,
        float maxX,
        float minZ,
        float maxZ
    )
    {
        // Position slightly on top of the floor
        float y = floorBounds.max.y + SPAWN_HEIGHT_OFFSET;

        // Get a position for the dragon within the opposite side of the cave
        // avoiding columns
        Vector3[] positions = new Vector3[numberOfPositions];
        int foundPositions = 0;

        while (foundPositions < numberOfPositions)
        {
            float x = Random.Range(minX, maxX);
            float z = Random.Range(minZ, maxZ);

            Vector3 position = new Vector3(x, y, z);

            // Keep sampling positions until you find one
            // that is not overlapping a column. Also, avoid spawning
            // overlapping agent
            if (!Physics.CheckSphere(position, SAFE_SPAWN_RADIUS, spawnBlockerLayers))
            {
                positions[foundPositions] = position;
                foundPositions++;
            }
        }

        return positions;
    }

    // ========================================================================
    // Utilities for agents to get information about the dungeon
    // ========================================================================
    public float GetAgentDensityWithinRadius(int agentId, Vector3 position, float radius)
    {
        int count = agents.Count(a =>
            a.Id != agentId && Vector3.Distance(a.transform.position, position) < radius
        );

        return count / (numberOfAgents - 1);
    }

    public float GetAgentDistanceFromTeam(int agentId, Vector3 position)
    {
        // Calc mean position of the agents in the team
        // excluded the one that is requesting the distance
        Vector3 sum = new Vector3(0, 0, 0);
        foreach (AgentBehavior other in agents)
        {
            if (agentId != other.Id)
            {
                sum += other.transform.position;
            }
        }

        Vector3 teamCentroid = sum / (agents.Count() - 1);

        return Vector3.Distance(teamCentroid, position);
    }

    public float NormalizedDistanceFromExit(Vector3 position)
    {
        // Distance between agent and door
        float distance = Vector3.Distance(position, door.transform.position);
        return distance / maxDistance;
    }

    private float ComputeUrgencyCausedByDragons()
    {
        // Return a value that represents how close the dragons are
        // to their cave
        float maxUrgency = 0f;

        foreach (var dragon in dragons)
        {
            if (dragon.isActiveAndEnabled)
            {
                float distanceFromCave = Vector3.Distance(
                    dragon.transform.position,
                    cave.transform.position
                );
                float urgency = 1 - (distanceFromCave / maxDistance);

                if (urgency > maxUrgency)
                {
                    maxUrgency = urgency;
                }
            }
        }

        // If close to 0, dragons are far from cave. If close to 1,
        // dragons are close to cave and about to escape
        return maxUrgency;
    }

    private float ComputeUrgencyCausedByTimer()
    {
        // Based on how much time is left to grab the key and escape, return
        // a value between 0 and 1
        return 1f - (timer.TimeLeft() / timer.maxTime);
    }

    // ========================================================================
    // General utilities
    // ========================================================================

    private void ChangeLightsColor(LightColor color)
    {
        Color newColor;
        ColorUtility.TryParseHtmlString(LightColorToHex[color], out newColor);

        // Change color of each column light
        foreach (Transform column in columns.transform)
        {
            // Get light game object
            Light light = column.transform.GetChild(1).GetComponent<Light>();

            light.color = newColor;
        }
    }

    private string GetTeamName()
    {
        return string.Join("_", personalitySettings.Select(p => p.personality.personalityName));
    }

    // ========================================================================
    // Stats
    // ========================================================================
    void FixedUpdate()
    {
        // Compute urgency value
        urgency =
            remainingDragons > 0 ? ComputeUrgencyCausedByDragons() : ComputeUrgencyCausedByTimer();

        if (!computeEpisodeStats)
            return;

        globalEpisodeStats.UpdateTimedStats();
    }
}
