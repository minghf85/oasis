# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
import sqlite3

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from graph_utils import get_dpeth, get_subgraph_by_time, plot_graph_like_tree


class prop_graph:

    def __init__(self, source_post_content, db_path="", viz=False):
        # Source tweet content for propagation
        self.source_post_content = source_post_content
        self.db_path = db_path  # Path to the db file obtained after simulation
        self.viz = viz  # Whether to visualize the result
        # Determine if the simulation ran successfully, False if the db
        # is empty
        self.post_exist = False

    def build_graph(self):
        # Connect to the SQLite database
        conn = sqlite3.connect(self.db_path)

        # Execute SQL query and load the results into a DataFrame
        query = "SELECT * FROM post"
        df = pd.read_sql(query, conn)
        conn.close()

        if df.empty:
            self.post_exist = False
            return

        # Find the source post (original post without original_post_id)
        source_posts = df[df['original_post_id'].isnull()]
        source_post = None

        # Find the source post that matches our content
        for _, post in source_posts.iterrows():
            if self.source_post_content[0:10] in str(post['content']):
                source_post = post
                self.post_exist = True
                self.root_id = str(post['user_id'])
                break

        if not self.post_exist:
            return

        # Create a directed graph
        self.G = nx.DiGraph()

        # Add root node with timestamp 0
        self.G.add_node(self.root_id, timestamp=0)

        # Create mapping from post_id to user_id and timestamp
        post_to_user = {}
        post_to_time = {}
        user_to_time = {}

        start_time = source_post['created_at']

        for _, row in df.iterrows():
            post_id = row['post_id']
            user_id = str(row['user_id'])
            created_at = row['created_at']

            post_to_user[post_id] = user_id
            post_to_time[post_id] = created_at

            # Calculate time difference from source post
            time_diff = created_at - start_time

            # Store the earliest time for each user
            if user_id not in user_to_time:
                user_to_time[user_id] = time_diff
            else:
                user_to_time[user_id] = min(user_to_time[user_id], time_diff)

        # Build the graph by connecting reposts to their original posts
        for _, row in df.iterrows():
            original_post_id = row['original_post_id']
            current_user = str(row['user_id'])
            current_time = row['created_at']

            # Skip if this is the original post
            if pd.isnull(original_post_id):
                continue

            # Find the user who posted the original post
            if original_post_id in post_to_user:
                original_user = post_to_user[original_post_id]

                # Add current user node if not exists
                if current_user not in self.G:
                    time_diff = current_time - start_time
                    self.G.add_node(current_user, timestamp=time_diff)

                # Add original user node if not exists
                if original_user not in self.G:
                    original_time = post_to_time[original_post_id]
                    time_diff = original_time - start_time
                    self.G.add_node(original_user, timestamp=time_diff)

                # Add edge from original user to current user
                self.G.add_edge(original_user, current_user)

        # Get the start and end timestamps of propagation
        self.start_timestamp = 0
        timestamps = nx.get_node_attributes(self.G, "timestamp")
        try:
            self.end_timestamp = max(timestamps.values()) + 3 if timestamps else 3
        except Exception as e:
            print(self.source_post_content)
            print(f"ERROR: {e}, may be caused by empty repost path")
            print(f"the simulation db is empty: {not self.post_exist}")

        # Calculate propagation graph depth, scale, maximum width
        # (max_breadth), and total structural virality
        self.total_depth = get_dpeth(self.G, source=self.root_id)
        self.total_scale = self.G.number_of_nodes()
        self.total_max_breadth = 0
        last_breadth_list = [1]
        for depth in range(self.total_depth):
            breadth = len(
                list(
                    nx.bfs_tree(
                        self.G, source=self.root_id, depth_limit=depth +
                        1).nodes())) - sum(last_breadth_list)
            last_breadth_list.append(breadth)
            if breadth > self.total_max_breadth:
                self.total_max_breadth = breadth

        undirect_G = self.G.to_undirected()
        self.total_structural_virality = nx.average_shortest_path_length(
            undirect_G)

    def viz_graph(self, time_threshold=10000):
        # Visualize the graph, can choose to only view the propagation graph
        # within the first time_threshold seconds
        subG = get_subgraph_by_time(self.G, time_threshold)
        plot_graph_like_tree(subG, self.root_id)

    def plot_depth_time(self, separate_ratio: float = 1):
        """
        Entire propagation process
        Detailed depiction of the data for the process before separate_ratio
        Rough depiction of the data afterwards
        Default to 1
        Use this parameter when the propagation time is very long, can be set
        to 0.01
        """
        # Calculate depth-time information
        depth_list = []
        # Normal interval is 1 for the time list, depth-time information needs
        # to be detailed enough
        self.d_t_list = list(
            range(int(self.start_timestamp), int(self.end_timestamp), 1))
        depth = 0
        for t in self.d_t_list:
            if depth < self.total_depth:
                try:
                    sub_g = get_subgraph_by_time(self.G, time_threshold=t)
                    depth = get_dpeth(sub_g, source=self.root_id)
                except Exception:
                    import pdb

                    pdb.set_trace()
            depth_list.append(depth)
        self.depth_list = depth_list

        if self.viz:
            # Use plot() function to draw a line chart
            _, ax = plt.subplots()
            ax.plot(self.d_t_list, self.depth_list)

            # Add titles and labels
            plt.title("Propagation depth-time")
            plt.xlabel("Time/minute")
            plt.ylabel("Depth")

            # Display the figure
            plt.show()
        else:
            return self.d_t_list, self.depth_list

    def plot_scale_time(self, separate_ratio: float = 1.0):
        """
        Detailed depiction of the data between the start and separate_ratio*T
        of the entire propagation process
        Rough depiction of the data afterwards
        Default to 1
        Use this parameter when the propagation time is very long, can be set
        to 0.1
        """
        self.node_nums = []
        # Detailed depiction of the data from start_time to separate point,
        # rough depiction from separate point to end_time
        separate_point = int(
            int(self.start_timestamp) + separate_ratio *
            (int(self.end_timestamp) - int(self.start_timestamp)))

        self.s_t_list = list(
            range(
                int(self.start_timestamp), separate_point,
                1))  # + list(range(separate_point, int(self.end_time), 1000))
        for t in self.s_t_list:
            try:
                sub_g = get_subgraph_by_time(self.G, time_threshold=t)
                node_num = sub_g.number_of_nodes()
            except Exception:
                import pdb

                pdb.set_trace()

            self.node_nums.append(node_num)

        if self.viz:
            # Use plot() function to draw a line chart
            _, ax = plt.subplots()
            ax.plot(self.s_t_list, self.node_nums)
            # Set the x-axis to log scale
            # ax.set_xscale('log')

            # Set the x-axis tick positions
            # ax.set_xticks([1, 10, 100, 1000, 10000])

            # Set the x-axis tick labels
            # ax.set_xticklabels(['1', '10', '100', '1k', '10k'])

            # Add titles and labels
            plt.title("Propagation scale-time")
            plt.xlabel("Time/minute")
            plt.ylabel("Scale")

            # Display the figure
            plt.show()
        else:
            return self.s_t_list, self.node_nums

    def plot_max_breadth_time(self, interval=1):
        self.max_breadth_list = []

        self.b_t_list = list(
            range(int(self.start_timestamp), int(self.end_timestamp),
                  interval))
        for t in self.b_t_list:
            try:
                sub_g = get_subgraph_by_time(self.G, time_threshold=t)
            except Exception:
                import pdb

                pdb.set_trace()
            max_depth = self.depth_list[t - self.b_t_list[0]]
            max_breadth = 0
            last_breadth_list = [1]
            for depth in range(max_depth):
                breadth = len(
                    list(
                        nx.bfs_tree(
                            sub_g, source=self.root_id, depth_limit=depth +
                            1).nodes())) - sum(last_breadth_list)
                last_breadth_list.append(breadth)
                if breadth > max_breadth:
                    max_breadth = breadth
            self.max_breadth_list.append(max_breadth)

        if self.viz:
            # Use plot() function to draw a line chart
            _, ax = plt.subplots()
            ax.plot(self.b_t_list, self.max_breadth_list)

            # Add titles and labels
            plt.title("Propagation max breadth-time")
            plt.xlabel("Time/minute")
            plt.ylabel("Max breadth")

            # Display the figure
            plt.show()
        else:
            return self.b_t_list, self.max_breadth_list

    def plot_structural_virality_time(self, interval=1):
        self.sv_list = []
        self.sv_t_list = list(
            range(int(self.start_timestamp), int(self.end_timestamp),
                  interval))

        for t in self.sv_t_list:
            try:
                sub_g = get_subgraph_by_time(self.G, time_threshold=t)
            except Exception:
                import pdb

                pdb.set_trace()
            sub_g = sub_g.to_undirected()
            sv = nx.average_shortest_path_length(sub_g)
            self.sv_list.append(sv)

        if self.viz:
            # Use plot() function to draw a line chart
            _, ax = plt.subplots()
            ax.plot(self.sv_t_list, self.sv_list)

            # Add titles and labels
            plt.title("Propagation structural virality-time")
            plt.xlabel("Time/minute")
            plt.ylabel("Structural virality")

            # Display the figure
            plt.show()
        else:
            return self.sv_t_list, self.sv_list