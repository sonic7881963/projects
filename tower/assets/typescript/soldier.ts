
import { _decorator, Component, Node, Collider, Contact2DType, Collider2D, IPhysics2DContact, ITriggerEvent, BoxCollider, BoxCollider2D, PhysicsSystem2D, Animation, Sprite } from 'cc';
import Gloabl from './gloabl';
import { hero } from './hero';
import { mainScene } from './mainScene';
const { ccclass, property } = _decorator;

 
@ccclass('soldier')
export class soldier extends Component {

    
    @property
    hp = 0;

    @property
    fight = 0;

    selfCollider: Collider2D;
    otherCollider: Collider2D;

    private speed: number = 100;

    private num: number = 0;

    private war: Boolean = false;

    onLoad(){
        let box = this.node.getComponent(BoxCollider2D);
        if(this.node.getComponent(Sprite).spriteFrame.name == "soldier"){
            this.hp = 20;
            this.fight = 5;
            box.tag = 1
        }else if(this.node.getComponent(Sprite).spriteFrame.name == "heroHz"){
            this.hp = 5;
            this.fight = 10;
            box.tag = 2
        }else if(this.node.getComponent(Sprite).spriteFrame.name == "heroGy"){
            this.hp = 30;
            this.fight = 20; 
            box.tag = 3
        }
    }

    start () {
        // [3]
        let collider = this.node.getComponent(BoxCollider2D);
        if(collider){
            let box = this.node.getComponent(BoxCollider2D);
            if(this.node.name == "hero"){
               box.group = 2;
            }else{
               box.group = 4;
            }
            collider.on(Contact2DType.BEGIN_CONTACT, this.onBeginContact, this);
            if (PhysicsSystem2D.instance) {
                PhysicsSystem2D.instance.on(Contact2DType.BEGIN_CONTACT, this.onBeginContact, this);
            }
        }
        
    }

    update (deltaTime: number) {
        if(this.node.name == "hero"){
            if(this.hp > 0){
                if(this.speed){
                    if (mainScene.js == true && this.war != true) {
                        this.speed = 150;
                    }
                    this.node.setPosition(this.node.position.x + this.speed * deltaTime , this.node.position.y);
                }else{
                    if (mainScene.js == true && this.war != true) {
                        Gloabl.speed = 150;
                    }
                    this.node.setPosition(this.node.position.x + Gloabl.speed * deltaTime , this.node.position.y);
                }
            }
            
        }else{
            if(this.hp > 0){
                if(this.speed){
                    if (mainScene.js == true && this.war != true) {
                        this.speed = 150;
                    }
                    this.node.setPosition(this.node.position.x - this.speed * deltaTime, this.node.position.y);
                }else{
                    if (mainScene.js == true && this.war != true) {
                        Gloabl.speed = 150;
                    }
                    this.node.setPosition(this.node.position.x - Gloabl.speed * deltaTime, this.node.position.y);
                }
            }
        }
        
    }


    onBeginContact (selfCollider: Collider2D, otherCollider: Collider2D, contact: IPhysics2DContact | null) {
        // 只在两个碰撞体开始接触时被调用一次
        console.log('onBeginContact');
        if(selfCollider.node.name == "box" || otherCollider.node.name == "box"){
            mainScene.Instance.winNode.active = true;
            return;
        }
        this.war = true;
        
        selfCollider.node.getComponent(soldier).speed = 0;
        otherCollider.node.getComponent(soldier).speed = 0;
        
        if(selfCollider.node.name == otherCollider.node.name){
            return;
        }
        this.selfCollider = selfCollider;
        this.otherCollider = otherCollider;
        Gloabl.speed = 0;

        let self_animationComponent = selfCollider.node.getComponent(Animation);
        self_animationComponent.play(self_animationComponent.clips[2].name);
        let other_animationComponent = otherCollider.node.getComponent(Animation);
        other_animationComponent.play(other_animationComponent.clips[2].name);
        if (mainScene.js == true) {
            self_animationComponent.getState(self_animationComponent.clips[2].name).speed = 1.5;
            other_animationComponent.getState(other_animationComponent.clips[2].name).speed = 1.5;
        }
        
       
        this.num = 0;
       
    }

    aniFightEnd(){
    
       
        this.selfCollider.node.getComponent(soldier).hp -= this.otherCollider.node.getComponent(soldier).fight;
        this.otherCollider.node.getComponent(soldier).hp -= this.selfCollider.node.getComponent(soldier).fight;
            
        
        console.log(this.selfCollider.node.getComponent(soldier).hp + " " +  this.otherCollider.node.getComponent(soldier).hp );

        let self_animationComponent = this.selfCollider.node.getComponent(Animation);
        let other_animationComponent = this.otherCollider.node.getComponent(Animation);
        if (this.selfCollider.node.getComponent(soldier).hp <= 0 && this.otherCollider.node.getComponent(soldier).hp <= 0){
            this.selfCollider.enabled = false;
            this.otherCollider.enabled = false;
            self_animationComponent.play(self_animationComponent.clips[1].name);
            other_animationComponent.play(other_animationComponent.clips[1].name);
            if (mainScene.js == true) {
                self_animationComponent.getState(self_animationComponent.clips[1].name).speed = 1.5;
                other_animationComponent.getState(other_animationComponent.clips[1].name).speed = 1.5;
            } 
        }else if(this.selfCollider.node.getComponent(soldier).hp <= 0){
            
            this.selfCollider.enabled = false;
            //this.selfCollider.destroy();
            self_animationComponent.play(self_animationComponent.clips[1].name);
            other_animationComponent.play(other_animationComponent.clips[0].name);
            this.war = false;
            if (mainScene.js == true) {
                self_animationComponent.getState(self_animationComponent.clips[1].name).speed = 1.5;
                other_animationComponent.getState(other_animationComponent.clips[0].name).speed = 1.5;
            }
        }else if(this.otherCollider.node.getComponent(soldier).hp <= 0){
            this.otherCollider.enabled = false;
            other_animationComponent.play(other_animationComponent.clips[1].name); 
            self_animationComponent.play(self_animationComponent.clips[0].name);
            this.war = false
            //this.otherCollider.destroy();
            if (mainScene.js == true) {
                self_animationComponent.getState(self_animationComponent.clips[1].name).speed = 1.5;
                other_animationComponent.getState(other_animationComponent.clips[0].name).speed = 1.5;
            }
        }else{
           
            self_animationComponent.play(self_animationComponent.clips[2].name);
            other_animationComponent.play(other_animationComponent.clips[2].name);
            if (mainScene.js == true) {
                self_animationComponent.getState(self_animationComponent.clips[2].name).speed = 1.5;
                other_animationComponent.getState(other_animationComponent.clips[2].name).speed = 1.5;
            }
        }
      ;
    }

    aniDeathEnd(){
        let self_animationComponent = this.selfCollider.node.getComponent(Animation);
        let other_animationComponent = this.otherCollider.node.getComponent(Animation);
        if(this.selfCollider.node.getComponent(soldier).hp <= 0){
            if(this.selfCollider.node.name == "enemy"){
                if(this.selfCollider.node.getComponent(BoxCollider2D).tag == 1){
                    Gloabl.coin += 5;
                }else if(this.selfCollider.node.getComponent(BoxCollider2D).tag == 2){
                    Gloabl.coin += 10;
                }else if(this.selfCollider.node.getComponent(BoxCollider2D).tag == 3){
                    Gloabl.coin += 30;
                }
            }
           
            this.selfCollider.node.active = false;
            this.selfCollider.node.removeFromParent();
           
            this.speed = 100;
            Gloabl.speed = 100;
            mainScene.Instance.refresh();
        }
         
        if(this.otherCollider.node.getComponent(soldier).hp <= 0){
            this.speed = 100;
            Gloabl.speed = 100;
            if(this.otherCollider.node.name == "enemy"){
                if(this.otherCollider.node.getComponent(BoxCollider2D).tag == 1){
                    Gloabl.coin += 5;
                }else if(this.otherCollider.node.getComponent(BoxCollider2D).tag == 2){
                    Gloabl.coin += 10;
                }else if(this.otherCollider.node.getComponent(BoxCollider2D).tag == 3){
                    Gloabl.coin += 30;
                }
            }
            this.otherCollider.node.active = false;
            this.otherCollider.node.removeFromParent();
          
            
            mainScene.Instance.refresh();
        }
        // else{
        //     self_animationComponent.play(self_animationComponent.clips[2].name);
        //     other_animationComponent.play(other_animationComponent.clips[2].name);
        // }
    }
}

/**
 * [1] Class member could be defined like this.
 * [2] Use `property` decorator if your want the member to be serializable.
 * [3] Your initialization goes here.
 * [4] Your update function goes here.
 *
 * Learn more about scripting: https://docs.cocos.com/creator/3.4/manual/zh/scripting/
 * Learn more about CCClass: https://docs.cocos.com/creator/3.4/manual/zh/scripting/ccclass.html
 * Learn more about life-cycle callbacks: https://docs.cocos.com/creator/3.4/manual/zh/scripting/life-cycle-callbacks.html
 */
