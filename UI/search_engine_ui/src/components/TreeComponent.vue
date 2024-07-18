<template>
  <div class="tree">
    <h2>Binary Tree</h2>
    <TreeNodeComponent v-if="tree" :node="tree" :level="0" />
    <div v-else>Loading...</div>
  </div>
</template>

<script>
import axios from 'axios';
import TreeNodeComponent from './TreeNodeComponent.vue';

export default {
  name: 'TreeComponent',
  components: {
    TreeNodeComponent
  },
  data() {
    return {
      tree: null
    };
  },
  created() {
    this.fetchTree();
  },
  methods: {
    async fetchTree() {
      try {
        const response = await axios.get('http://localhost:5000/get_tree');
        this.tree = response.data;
      } catch (error) {
        console.error('Error fetching tree:', error);
      }
    }
  }
};
</script>

<style scoped>
.tree {
  font-family: Arial, sans-serif;
  margin: 20px;
  display: flex;
  flex-direction: column;
  align-items: center;
}

.tree h2 {
  margin-bottom: 20px;
}
</style>