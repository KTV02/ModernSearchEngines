<template>
  <div :class="['tree-node', levelClass]">
    <div class="node-content">
      <span>{{ node.word }} ({{ node.count }})</span>
    </div>
    <div v-if="node.left || node.right" class="children">
      <TreeNodeComponent v-if="node.left" :node="node.left" :level="level + 1" class="left-child" />
      <TreeNodeComponent v-if="node.right" :node="node.right" :level="level + 1" class="right-child" />
    </div>
  </div>
</template>

<script>
export default {
  name: 'TreeNodeComponent',
  props: {
    node: Object,
    level: {
      type: Number,
      default: 0
    }
  },
  computed: {
    levelClass() {
      return `level-${this.level}`;
    }
  }
};
</script>

<style scoped>
.tree-node {
  display: flex;
  flex-direction: column;
  align-items: center;
  margin: 10px 0;
}

.node-content {
  display: inline-block;
  padding: 8px 12px;
  background-color: #f5f5f5;
  border: 1px solid #ccc;
  border-radius: 4px;
  margin-bottom: 10px;
  text-align: center;
}

.children {
  display: flex;
  justify-content: space-between;
  width: 100%;
}

.left-child {
  border-left: 1px solid #ccc;
  padding-left: 10px;
}

.right-child {
  border-right: 1px solid #ccc;
  padding-right: 10px;
}

/* Level-specific colors */
.level-0 .node-content {
  background-color: #f0f8ff;
}

.level-1 .node-content {
  background-color: #e6f2ff;
}

.level-2 .node-content {
  background-color: #ccddff;
}

.level-3 .node-content {
  background-color: #b3c7ff;
}

/* Add more levels if necessary */
</style>