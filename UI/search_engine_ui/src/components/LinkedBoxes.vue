<template>
  <div class="linked-boxes">
    <div class="center-dot"></div>
    <div v-for="(box, index) in boxes" :key="index" :class="'box box' + (index + 1)">
      <div class="box-name">{{ box.name }}</div>
      <div class="links">
        <div v-for="(link, linkIndex) in box.urls" :key="linkIndex" class="link">
          <a :href="link.url" target="_blank">{{ link.title }}</a>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import axios from 'axios';

export default {
  name: 'LinkedBoxes',
  data() {
    return {
      boxes: []
    };
  },
  created() {
    this.fetchLinks();
  },
  methods: {
    async fetchLinks() {
      try {
        const response = await axios.get('http://localhost:5000/get_links');
        this.boxes = response.data;
      } catch (error) {
        console.error('Error fetching links:', error);
      }
    }
  }
};
</script>

<style scoped>
.linked-boxes {
  position: relative;
  width: 400px;
  height: 400px;
  margin: 0 auto;
  display: flex;
  justify-content: center;
  align-items: center;
}

.center-dot {
  width: 10px;
  height: 10px;
  background-color: black;
  border-radius: 50%;
  position: absolute;
}

.box {
  position: absolute;
  width: 120px;
  height: 120px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  text-align: center;
  border: 1px solid #ccc;
  border-radius: 8px;
  background-color: #f9f9f9;
  padding: 10px;
}

.box-name {
  font-weight: bold;
  margin-bottom: 10px;
}

.links {
  display: flex;
  flex-direction: column;
}

.link {
  margin: 5px 0;
}

.box1 {
  top: 0;
  left: 50%;
  transform: translate(-50%, -100%);
}

.box2 {
  top: 50%;
  right: 0;
  transform: translate(100%, -50%);
}

.box3 {
  bottom: 0;
  left: 50%;
  transform: translate(-50%, 100%);
}

.box4 {
  top: 50%;
  left: 0;
  transform: translate(-100%, -50%);
}

.box5 {
  bottom: 0;
  right: 0;
  transform: translate(100%, 100%);
}
</style>