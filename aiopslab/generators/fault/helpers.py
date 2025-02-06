import subprocess

# SocialNetwork service process names
sn_svc_process_names = [
    "ComposePostServ",
    "HomeTimelineSer",
    "MediaService",
    "PostStorageServ",
    "SocialGraphServ",
    "TextService",
    "UserService",
    "UrlShortenServi",
    "UserMentionServ",
    "UserTimelineSer",
    "UniqueIdService",
]

# SocialNetwork MongoDB process names
sn_mongod_process_names = ["mongod"]

# SocialNetwork Redis process names
sn_redis_process_names = ["redis-server"]

# SocialNetwork Memcached process names
sn_memcached_process_names = ["memcached"]

# HotelResearvation service process names
hr_svc_process_names = [
    "geo",
    "frontend",
    "consul",
    "profile",
    "rate",
    "recommendation",
    "reservation",
    "search",
    "user",
]

# HotelResearvation MongoDB process names
hr_mongod_process_names = ["mongod"]

# HotelResearvation Memcached process names
hr_memcached_process_names = ["memcached"]


def get_pids_by_name_contain(search_term):
    """
    Get a list of PIDs for processes whose command contains the given search term.

    :param search_term: The term to search for in process names (case-sensitive).
    :return: A list of PIDs (integers) matching the search term.
    """
    try:
        result = subprocess.run(
            ["ps", "-e", "-o", "pid,comm"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Error running ps command: {result.stderr.strip()}")

        # Filter the output for lines containing the search term
        matching_pids = []
        for line in result.stdout.splitlines():
            if search_term in line:
                parts = line.split(maxsplit=1)
                if parts:  # Ensure we have at least one part
                    pid = parts[0]
                    if pid.isdigit():
                        matching_pids.append(int(pid))

        return matching_pids

    except Exception as e:
        print(f"Error: {e}")
        return []


def get_pids_by_name(search_term):
    """
    Get a list of PIDs for processes whose command exactly match the given search term.

    :param search_term: The term to search for in process names (case-sensitive).
    :return: A list of PIDs (integers) matching exactly the search term.
    """
    try:
        result = subprocess.run(
            ["ps", "-e", "-o", "pid,comm"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Error running ps command: {result.stderr.strip()}")

        # Filter the output for lines containing the search term
        matching_pids = []
        for line in result.stdout.splitlines():
            parts = line.split(maxsplit=1)
            if len(parts) == 2:  # Ensure we have both PID and command
                pid, command = parts
                if command == search_term:  # Exact match check
                    if pid.isdigit():
                        matching_pids.append(int(pid))

        return matching_pids

    except Exception as e:
        print(f"Error: {e}")
        return []


if __name__ == "__main__":
    search_term = "HomeTimelineSer"
    pids = get_pids_by_name(search_term)
    print(f"Processes with '{search_term}' in their name: {pids}")
